#!/usr/bin/env python3

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

COMPRESSORS = ("sdp4bit", "zfp")
EXPECTED_PERFORMANCE = {name: 80 for name in COMPRESSORS}
EXPECTED_SMOKE = {name: 33 for name in COMPRESSORS}
EXPECTED_MEMORY = {name: 13 for name in COMPRESSORS}
EXPECTED_LAYOUT = 6


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0],
        "algorithm": parts[1],
        "compressor": parts[2],
        "depth": int(parts[3][1:]),
        "bytes": int(parts[4][1:]),
    }


def row_key(row):
    return (
        row["collective"], row["algorithm"], row["compressor"],
        int(row["depth"]), int(row["bytes"]), int(row["inplace"]),
    )


def parse_benchmark(path, case):
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            base = {**case, "bytes": int(tokens[0])}
            rows.append({
                **base, "inplace": 0, "time_us": float(tokens[5]),
                "algbw_gbps": float(tokens[6]),
                "busbw_gbps": float(tokens[7]),
            })
            rows.append({
                **base, "inplace": 1, "time_us": float(tokens[9]),
                "algbw_gbps": float(tokens[10]),
                "busbw_gbps": float(tokens[11]),
            })
    return rows


def write_performance(result_root, history_root):
    rows = []
    counts = {}
    for compressor in COMPRESSORS:
        raw = result_root / compressor / "raw" / "performance"
        markers = sorted(raw.glob("*.ok"))
        counts[compressor] = len(markers)
        for marker in markers:
            rows.extend(parse_benchmark(marker.with_suffix(".log"),
                                        case_fields(marker)))

    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "time_us", "algbw_gbps", "busbw_gbps",
    ]
    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    baseline = {}
    with (history_root / "M9" / "performance.csv").open() as stream:
        for row in csv.DictReader(stream):
            baseline[row_key(row)] = float(row["time_us"])

    comparisons = []
    failures = []
    missing = []
    for row in rows:
        old_us = baseline.get(row_key(row))
        if old_us is None:
            missing.append(row_key(row))
            continue
        limit_us = old_us * 1.03
        passed = row["time_us"] <= limit_us
        comparison = {
            **{name: row[name] for name in (
                "collective", "algorithm", "compressor", "depth",
                "bytes", "inplace")},
            "m9_us": old_us,
            "m10_us": row["time_us"],
            "limit_us": limit_us,
            "relative_delta": (row["time_us"] - old_us) / old_us,
            "passed": int(passed),
        }
        comparisons.append(comparison)
        if not passed:
            failures.append(comparison)

    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "m9_us", "m10_us", "limit_us", "relative_delta",
        "passed",
    ]
    with (result_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return rows, counts, comparisons, failures, missing


def write_paired_performance(result_root):
    comparisons = []
    for compressor in COMPRESSORS:
        m9_raw = result_root / "paired" / "m9" / compressor / "raw" / "performance"
        m10_raw = result_root / "paired" / "m10" / compressor / "raw" / "performance"
        for marker in sorted(m10_raw.glob("*.ok")):
            case = case_fields(marker)
            m9_log = m9_raw / marker.with_suffix(".log").name
            if not m9_log.exists():
                continue
            baseline = {
                row_key(row): row["time_us"]
                for row in parse_benchmark(m9_log, case)
            }
            for row in parse_benchmark(marker.with_suffix(".log"), case):
                old_us = baseline.get(row_key(row))
                if old_us is None:
                    continue
                limit_us = old_us * 1.03
                comparisons.append({
                    **{name: row[name] for name in (
                        "collective", "algorithm", "compressor", "depth",
                        "bytes", "inplace")},
                    "m9_us": old_us,
                    "m10_us": row["time_us"],
                    "limit_us": limit_us,
                    "relative_delta": (row["time_us"] - old_us) / old_us,
                    "passed": int(row["time_us"] <= limit_us),
                })

    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "m9_us", "m10_us", "limit_us", "relative_delta",
        "passed",
    ]
    with (result_root / "paired-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons, [row for row in comparisons if not row["passed"]]


def write_layout(result_root, history_root):
    rows = []
    raw = result_root / "layout" / "raw" / "layout"
    markers = sorted(raw.glob("*.ok"))
    for marker in markers:
        with marker.with_suffix(".log").open(errors="replace") as stream:
            for row in csv.DictReader(stream):
                rows.append({
                    "bytes": int(row["bytes"]),
                    "chunks": int(row["chunks"]),
                    "depth": int(row["depth"]),
                    "pack_us": float(row["pack_us"]),
                    "unpack_us": float(row["unpack_us"]),
                    "layout_us": float(row["layout_us"]),
                })
    fields = ["bytes", "chunks", "depth", "pack_us", "unpack_us",
              "layout_us"]
    with (result_root / "layout.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    baseline = {}
    with (history_root / "M8" / "layout.csv").open() as stream:
        for row in csv.DictReader(stream):
            baseline[(int(row["bytes"]), int(row["depth"]))] = float(
                row["layout_us"])
    comparisons = []
    failures = []
    for row in rows:
        old_us = baseline.get((row["bytes"], row["depth"]))
        if old_us is None:
            failures.append(row)
            continue
        passed = row["layout_us"] <= old_us * 1.05
        comparison = {
            "bytes": row["bytes"], "depth": row["depth"],
            "m8_layout_us": old_us, "m10_layout_us": row["layout_us"],
            "relative_delta": (row["layout_us"] - old_us) / old_us,
            "passed": int(passed),
        }
        comparisons.append(comparison)
        if not passed:
            failures.append(comparison)
    fields = ["bytes", "depth", "m8_layout_us", "m10_layout_us",
              "relative_delta", "passed"]
    with (result_root / "layout-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return rows, len(markers), comparisons, failures


ARENA_FILES = {
    "alltoall": ("arena-alltoall.csv", "pipeline"),
    "allgather": ("arena-allgather.csv", "pipeline"),
    "reducescatter": ("arena-reducescatter.csv", "oneshot"),
    "allreduce": ("arena-allreduce.csv", None),
}


def read_arena_workspaces(result_root):
    workspaces = {}
    for collective, (filename, default_algorithm) in ARENA_FILES.items():
        with (result_root / filename).open() as stream:
            for row in csv.DictReader(stream):
                algorithm = row.get("algorithm") or default_algorithm
                key = (collective, algorithm, int(row["bytes"]),
                       int(row["requested_depth"]))
                workspaces[key] = max(workspaces.get(key, 0),
                                      int(row["workspace_bytes"]))
    return workspaces


def peak_memory(path):
    peaks = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                peaks[fields[2]] = max(peaks[fields[2]], int(fields[3]))
    return sum(peaks.values()), len(peaks)


VMM_RELEASE = re.compile(
    r"COCCL VMM release .*remaining_virtual (\d+) "
    r"remaining_physical (\d+) remaining_registered (\d+)")


def expected_workspace(case):
    if case["collective"] in ("alltoall", "allgather"):
        return 2 * case["bytes"]
    if case["collective"] == "reducescatter":
        return 8 * case["bytes"]
    if case["collective"] == "allreduce" and case["algorithm"] == "twoshot":
        return 2 * case["bytes"]
    return None


def write_memory(result_root, history_root):
    arena = read_arena_workspaces(result_root)
    m9 = {}
    with (history_root / "M9" / "memory.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["collective"], row["algorithm"], row["compressor"],
                   int(row["depth"]), int(row["bytes"]))
            m9[key] = int(row["workspace_bytes_per_rank"])

    rows = []
    counts = {}
    failures = []
    for compressor in COMPRESSORS:
        raw = result_root / compressor / "raw" / "memory"
        markers = sorted(raw.glob("*.ok"))
        counts[compressor] = len(markers)
        for marker in markers:
            case = case_fields(marker)
            key = (case["collective"], case["algorithm"], case["bytes"],
                   case["depth"])
            workspace = arena.get(key, 0)
            expected = expected_workspace(case)
            formula_ok = expected is None or workspace == expected
            old_key = (case["collective"], case["algorithm"], compressor,
                       case["depth"], case["bytes"])
            old_workspace = m9.get(old_key, 0)
            decreased = old_workspace == 0 or workspace < old_workspace
            already_at_target = (
                old_workspace != 0 and expected is not None and
                old_workspace == expected)
            workspace_gate_ok = decreased or already_at_target
            log_text = marker.with_suffix(".log").read_text(errors="replace")
            releases = VMM_RELEASE.findall(log_text)
            release_ok = len(releases) >= 4 and all(
                all(int(value) == 0 for value in release)
                for release in releases)
            sample = result_root / compressor / "memory-samples" / (
                marker.stem + ".csv")
            peak, gpu_count = peak_memory(sample)
            row = {
                **case,
                "peak_total_mib": peak,
                "sampled_gpus": gpu_count,
                "m9_workspace_bytes_per_rank": old_workspace,
                "m10_workspace_bytes_per_rank": workspace,
                "expected_workspace_bytes_per_rank": expected or workspace,
                "workspace_reduction": (
                    (old_workspace - workspace) / old_workspace
                    if old_workspace else 0.0),
                "formula_ok": int(formula_ok),
                "decreased_vs_m9": int(decreased),
                "already_at_target": int(already_at_target),
                "vmm_release_ok": int(release_ok),
            }
            rows.append(row)
            if (gpu_count != 4 or not formula_ok or
                    not workspace_gate_ok or not release_ok):
                failures.append(row)

    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "peak_total_mib", "sampled_gpus",
        "m9_workspace_bytes_per_rank", "m10_workspace_bytes_per_rank",
        "expected_workspace_bytes_per_rank", "workspace_reduction",
        "formula_ok", "decreased_vs_m9", "already_at_target",
        "vmm_release_ok",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, counts, failures


def write_arena_layouts(result_root, memory_rows):
    unique = {}
    for row in memory_rows:
        key = (row["collective"], row["algorithm"], row["depth"],
               row["bytes"])
        unique[key] = row
    with (result_root / "arena_layouts.md").open("w") as stream:
        stream.write("# M10 Unified Arena Layouts\n\n")
        stream.write(
            "Each slice uses `max(s[0], max(s[i-1] + s[i]))`; even temps "
            "start at the left edge and odd temps at the right edge.\n\n")
        stream.write("| collective | algorithm | depth | bytes | M9 workspace | M10 workspace | reduction |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|\n")
        for key, row in sorted(unique.items()):
            old = row["m9_workspace_bytes_per_rank"]
            reduction = f"{row['workspace_reduction']:.2%}" if old else "n/a"
            stream.write(
                f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
                f"{old or '-'} | {row['m10_workspace_bytes_per_rank']} | "
                f"{reduction} |\n")


def main():
    result_root = Path(sys.argv[1])
    history_root = Path(sys.argv[2])
    performance, performance_counts, comparisons, performance_failures, missing = (
        write_performance(result_root, history_root))
    paired, paired_failures = write_paired_performance(result_root)
    paired_keys = {row_key(row) for row in paired}
    unresolved_performance = [
        row for row in performance_failures if row_key(row) not in paired_keys
    ] + paired_failures
    layout, layout_count, layout_comparisons, layout_failures = write_layout(
        result_root, history_root)
    memory, memory_counts, memory_failures = write_memory(
        result_root, history_root)
    write_arena_layouts(result_root, memory)

    smoke_counts = {
        name: len(list((result_root / name / "raw" / "smoke").glob("*.ok")))
        for name in COMPRESSORS
    }
    host_ok = "PASS" in (result_root / "unified-host.txt").read_text(
        errors="replace")
    layout_correct = "PASS" in (
        result_root / "layout-correctness.txt").read_text(errors="replace")
    complete = (
        performance_counts == EXPECTED_PERFORMANCE and
        smoke_counts == EXPECTED_SMOKE and
        memory_counts == EXPECTED_MEMORY and
        layout_count == EXPECTED_LAYOUT and
        len(performance) == 2 * sum(EXPECTED_PERFORMANCE.values()))
    passed = (complete and host_ok and layout_correct and
              not unresolved_performance and not missing and
              not layout_failures and not memory_failures)

    with (result_root / "report.md").open("w") as stream:
        stream.write("# M10 Unified Arena Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write("## Completeness\n\n")
        stream.write(f"- Performance processes: {performance_counts} / {EXPECTED_PERFORMANCE}.\n")
        stream.write(f"- Smoke processes: {smoke_counts} / {EXPECTED_SMOKE}.\n")
        stream.write(f"- Memory processes: {memory_counts} / {EXPECTED_MEMORY}.\n")
        stream.write(f"- Layout processes: {layout_count} / {EXPECTED_LAYOUT}.\n")
        stream.write(f"- Unified Host test: {'PASS' if host_ok else 'FAIL'}.\n")
        stream.write(f"- GPU layout correctness: {'PASS' if layout_correct else 'FAIL'}.\n\n")
        stream.write("## Performance\n\n")
        stream.write(f"- Compared M9 rows: {len(comparisons)}.\n")
        stream.write(f"- Historical M9 regressions above 3%: {len(performance_failures)}.\n")
        stream.write(f"- Paired M9/M10 comparison rows: {len(paired)}.\n")
        stream.write(f"- Paired regressions above 3%: {len(paired_failures)}.\n")
        stream.write(f"- Unresolved regressions above 3%: {len(unresolved_performance)}.\n")
        stream.write(f"- Missing M9 baselines: {len(missing)}.\n")
        stream.write(f"- Pack/Unpack regressions above 5%: {len(layout_failures)}.\n\n")
        stream.write("## Workspace\n\n")
        stream.write(f"- Memory/formula/release failures: {len(memory_failures)}.\n")
        stream.write("- A2A, AG and AR TwoShot target `2B`; RS OneShot target `2 * raw input = 8 * recv bytes` for four ranks.\n")
        stream.write("- An M9 fallback already equal to the target may remain equal; it is reported as `already_at_target`, not as a reduction.\n")
        stream.write("- Detailed physical capacities are in `arena_layouts.md` and `memory.csv`.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
