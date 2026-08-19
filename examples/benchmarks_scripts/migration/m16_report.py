#!/usr/bin/env python3

import csv
import collections
import math
import pathlib
import sys


FIELDS = (
    "topology", "rank_count", "operation", "algorithm", "compressor",
    "dtype", "depth", "output_elements", "mean_relative_error",
)
FLOORS = {"float": 1.0e-6, "half": 1.0e-3, "bfloat16": 1.0e-2}
SINGLE_OPERATIONS = (
    ("alltoall", "default"),
    ("allgather", "default"),
    ("reducescatter", "oneshot"),
    ("allreduce", "oneshot"),
    ("allreduce", "twoshot"),
    ("sendrecv", "default"),
)
HIERARCHICAL_OPERATIONS = (
    ("reducescatter", "twoshot"),
    ("allreduce", "tripleshot"),
)


def parse_line(line):
    if not line.startswith("COCCL_CORRECTNESS "):
        return None
    values = {}
    for field in line.strip().split()[1:]:
        key, value = field.split("=", 1)
        values[key] = value
    if any(field not in values for field in FIELDS):
        raise ValueError(f"incomplete correctness line: {line.strip()}")
    values["rank_count"] = int(values["rank_count"])
    values["depth"] = int(values["depth"])
    values["output_elements"] = int(values["output_elements"])
    values["mean_relative_error"] = float(values["mean_relative_error"])
    return values


def load_rows(root, implementation):
    rows = []
    for path in sorted((root / "raw" / implementation).glob("*/*.log")):
        for line in path.read_text(errors="replace").splitlines():
            row = parse_line(line)
            if row is not None:
                row["log"] = str(path)
                rows.append(row)
    return rows


def key(row):
    return tuple(row[field] for field in (
        "topology", "rank_count", "operation", "algorithm", "compressor",
        "dtype",
    ))


def expected_current():
    expected = set()
    for compressor in ("sdp4bit", "zfp", "dietgpu"):
        for datatype in FLOORS:
            for depth in (1, 2, 4, 8):
                for operation, algorithm in SINGLE_OPERATIONS:
                    expected.add(("single-node", 4, operation, algorithm,
                                  compressor, datatype, depth))
                for operation, algorithm in HIERARCHICAL_OPERATIONS:
                    expected.add(("2x4", 8, operation, algorithm, compressor,
                                  datatype, depth))
    for datatype in FLOORS:
        for depth in (1, 2, 4, 8):
            expected.add(("single-node", 4, "allgather", "subadd",
                          "sdp4bit", datatype, depth))
    return expected


def row_identity(row):
    return key(row) + (row["depth"],)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: m16_report.py RESULT_ROOT")
    root = pathlib.Path(sys.argv[1])
    baseline_rows = load_rows(root, "baseline")
    current_rows = load_rows(root, "current")
    baseline_by_key = {key(row): row for row in baseline_rows
                       if math.isfinite(row["mean_relative_error"])}
    current_depth_one = {key(row): row for row in current_rows
                         if row["depth"] == 1}

    evaluated = []
    failed = []
    for row in current_rows:
        error = row["mean_relative_error"]
        floor = FLOORS[row["dtype"]]
        reference = None
        reference_source = ""
        if row["compressor"] == "dietgpu":
            reduction = row["operation"] in ("allreduce", "reducescatter")
            limit = floor if reduction else 0.0
            passed = math.isfinite(error) and error <= limit
            reference_source = "lossless-contract"
        else:
            baseline = baseline_by_key.get(key(row))
            if baseline is not None:
                reference = baseline["mean_relative_error"]
                reference_source = "initial-coccl-depth1"
            else:
                depth_one = current_depth_one.get(key(row))
                if depth_one is not None:
                    reference = depth_one["mean_relative_error"]
                    reference_source = "migrated-depth1"
            limit = max(10.0 * reference, floor) if reference is not None else 0.0
            passed = reference is not None and math.isfinite(error) and error <= limit
        ratio = ""
        if reference is not None:
            ratio = error / reference if reference > 0.0 else (
                0.0 if error == 0.0 else math.inf)
        result = dict(row)
        result.update({
            "reference_error": "" if reference is None else reference,
            "reference_source": reference_source,
            "error_ratio": ratio,
            "error_limit": limit,
            "pass": int(passed),
        })
        evaluated.append(result)
        if not passed:
            failed.append(result)

    observed = {row_identity(row) for row in current_rows}
    missing = sorted(expected_current() - observed)
    duplicate_count = len(current_rows) - len(observed)
    status = not failed and not missing and duplicate_count == 0

    baseline_fields = list(FIELDS) + ["log"]
    correctness_fields = list(FIELDS) + [
        "reference_error", "reference_source", "error_ratio", "error_limit",
        "pass", "log"
    ]
    write_csv(root / "baseline_errors.csv", baseline_fields, baseline_rows)
    write_csv(root / "correctness.csv", correctness_fields, evaluated)
    failed_root = root / "failed_cases"
    failed_root.mkdir(exist_ok=True)
    write_csv(failed_root / "cases.csv", correctness_fields, failed)

    reference_counts = collections.Counter(
        row["reference_source"] for row in evaluated)
    finite_ratios = [row["error_ratio"] for row in evaluated
                     if row["error_ratio"] != "" and
                     math.isfinite(row["error_ratio"])]
    finite_baselines = sum(
        math.isfinite(row["mean_relative_error"]) for row in baseline_rows)
    dietgpu_rows = [row for row in current_rows
                    if row["compressor"] == "dietgpu"]
    dietgpu_non_reduction_max = max(
        row["mean_relative_error"] for row in dietgpu_rows
        if row["operation"] not in ("allreduce", "reducescatter"))
    dietgpu_reduction_max = max(
        row["mean_relative_error"] for row in dietgpu_rows
        if row["operation"] in ("allreduce", "reducescatter"))

    report = [
        "# M16 COCCL Correctness Report",
        "",
        f"Status: {'PASS' if status else 'FAIL'}",
        "",
        f"- Initial COCCL baseline rows: {len(baseline_rows)}.",
        f"- COCCL-migrate rows: {len(current_rows)}/{len(expected_current())}.",
        f"- Failed error comparisons: {len(failed)}.",
        f"- Missing matrix rows: {len(missing)}.",
        f"- Duplicate matrix rows: {duplicate_count}.",
        "- Every row uses all ranks and all output elements in the reported mean.",
        f"- Initial depth-1 references: "
        f"{reference_counts['initial-coccl-depth1']} rows.",
        f"- Migrated depth-1 fallback references: "
        f"{reference_counts['migrated-depth1']} rows.",
        f"- Lossless-contract checks: "
        f"{reference_counts['lossless-contract']} rows.",
        f"- Finite initial baselines: {finite_baselines}/{len(baseline_rows)}.",
        f"- Maximum finite migrated/reference ratio: "
        f"{max(finite_ratios):.6f}.",
        f"- dietGPU maximum non-reduction error: "
        f"{dietgpu_non_reduction_max:.6e}.",
        f"- dietGPU maximum reduction error: "
        f"{dietgpu_reduction_max:.6e}.",
        "- Migrated depth-1 is used only when the initial implementation has "
        "no finite result for that case.",
    ]
    if missing:
        report.extend(("", "## Missing", ""))
        report.extend(f"- {item}" for item in missing[:20])
    if failed:
        report.extend(("", "## Failed", ""))
        report.extend(
            f"- {row_identity(row)}: error={row['mean_relative_error']:.6e}, "
            f"limit={row['error_limit']:.6e}"
            for row in failed[:20]
        )
    (root / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
