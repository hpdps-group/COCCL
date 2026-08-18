#!/usr/bin/env python3

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COMPRESSORS = ("sdp4bit", "zfp")
EXPECTED_PERFORMANCE = {name: 238 for name in COMPRESSORS}
EXPECTED_SMOKE = {name: 5 for name in COMPRESSORS}
EXPECTED_MEMORY = {name: 12 for name in COMPRESSORS}
GATE_BYTES = 512 << 20


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0], "algorithm": parts[1],
        "compressor": parts[2], "depth": int(parts[3][1:]),
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


def read_performance(result_root):
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
    for name in ("full_benchmark.csv", "performance.csv"):
        with (result_root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows, counts


def load_rows(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        return {row_key(row): float(row["time_us"])
                for row in csv.DictReader(stream)}


def load_m9_formula_limits(path):
    limits = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (
                row["collective"], row["algorithm"], row["compressor"],
                int(row["depth"]), int(row["bytes"]), int(row["inplace"]),
            )
            limits[key] = float(row["limit_us"])
    return limits


def load_confirmations(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        return {
            (
                row["collective"], row["algorithm"], row["compressor"],
                int(row["depth"]), int(row["bytes"]), 0,
            ): row
            for row in csv.DictReader(stream)
        }


def compare_performance(result_root, history_root, rows):
    m9 = load_rows(history_root / "M9" / "performance.csv")
    m10 = load_rows(history_root / "M10" / "performance.csv")
    formula_limits = load_m9_formula_limits(
        history_root / "M9" / "comparison.csv")
    confirmations = load_confirmations(result_root / "confirmation.csv")
    comparisons = []
    failures = []
    initial_failures = 0
    confirmed = 0
    for row in rows:
        if row["inplace"] != 0 or row["bytes"] < GATE_BYTES:
            continue
        key = row_key(row)
        m9_us = m9.get(key)
        m10_us = m10.get(key)
        formula_limit = formula_limits.get(key)
        history_ok = (
            m9_us is not None and row["time_us"] <= 1.03 * m9_us and
            (m10_us is None or row["time_us"] <= 1.03 * m10_us))
        formula_ok = (formula_limit is not None and
                      row["time_us"] <= formula_limit)
        initial_passed = history_ok and formula_ok
        if not initial_passed:
            initial_failures += 1
        confirmation = confirmations.get(key)
        confirmation_us = None
        confirmation_baseline_us = None
        confirmation_kind = ""
        confirmation_passed = False
        if confirmation is not None:
            confirmation_us = float(confirmation["confirmation_us"])
            confirmation_kind = confirmation["kind"]
            confirmation_baseline_us = (
                float(confirmation["baseline_us"])
                if confirmation["baseline_us"] else None)
            confirmation_passed = (
                m9_us is not None and confirmation_us <= 1.03 * m9_us and
                formula_limit is not None and confirmation_us <= formula_limit)
            if confirmation_kind == "paired":
                confirmation_passed = (
                    confirmation_passed and
                    confirmation_baseline_us is not None and
                    confirmation_us <= 1.03 * confirmation_baseline_us)
            elif m10_us is not None:
                confirmation_passed = (
                    confirmation_passed and confirmation_us <= 1.03 * m10_us)
            if confirmation_passed:
                confirmed += 1
        passed = initial_passed or confirmation_passed
        comparison = {
            **{name: row[name] for name in (
                "collective", "algorithm", "compressor", "depth",
                "bytes", "inplace")},
            "m9_us": m9_us, "m10_us": m10_us,
            "m11_us": row["time_us"],
            "m9_limit_us": None if m9_us is None else 1.03 * m9_us,
            "m10_limit_us": None if m10_us is None else 1.03 * m10_us,
            "historical_formula_limit_us": formula_limit,
            "history_passed": int(history_ok),
            "formula_passed": int(formula_ok),
            "initial_passed": int(initial_passed),
            "confirmation_kind": confirmation_kind,
            "confirmation_us": confirmation_us,
            "confirmation_baseline_us": confirmation_baseline_us,
            "confirmation_passed": int(confirmation_passed),
            "passed": int(passed),
        }
        comparisons.append(comparison)
        if not comparison["passed"]:
            failures.append(comparison)
    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "m9_us", "m10_us", "m11_us", "m9_limit_us",
        "m10_limit_us", "historical_formula_limit_us",
        "history_passed", "formula_passed", "initial_passed",
        "confirmation_kind", "confirmation_us",
        "confirmation_baseline_us", "confirmation_passed", "passed",
    ]
    with (result_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons, failures, initial_failures, confirmed


def write_crossovers(result_root, history_root, rows):
    native = {}
    with (history_root / "M9" / "performance.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["compressor"] == "native" and int(row["inplace"]) == 0:
                native[(row["collective"], int(row["bytes"]))] = float(
                    row["time_us"])
    crossed = {}
    for row in rows:
        if row["inplace"] != 0:
            continue
        old = native.get((row["collective"], row["bytes"]))
        if old is not None and row["time_us"] <= old:
            key = (row["compressor"], row["collective"],
                   row["algorithm"], row["depth"])
            crossed[key] = min(crossed.get(key, row["bytes"]), row["bytes"])
    with (result_root / "crossover.csv").open("w", newline="") as stream:
        fields = ["compressor", "collective", "algorithm", "depth",
                  "crossover_bytes"]
        writer = csv.writer(stream)
        writer.writerow(fields)
        for key, size in sorted(crossed.items()):
            writer.writerow((*key, size))
    return crossed


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


def write_memory(result_root, history_root):
    baseline = {}
    with (history_root / "M10" / "memory.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["collective"], row["algorithm"], row["compressor"],
                   int(row["depth"]), int(row["bytes"]))
            baseline[key] = int(row["peak_total_mib"])
    rows = []
    counts = {}
    failures = []
    for compressor in COMPRESSORS:
        raw = result_root / compressor / "raw" / "memory"
        markers = sorted(raw.glob("*.ok"))
        counts[compressor] = len(markers)
        for marker in markers:
            case = case_fields(marker)
            sample = result_root / compressor / "memory-samples" / (
                marker.stem + ".csv")
            peak, gpu_count = peak_memory(sample)
            key = (case["collective"], case["algorithm"], compressor,
                   case["depth"], case["bytes"])
            old_peak = baseline.get(key)
            log = marker.with_suffix(".log").read_text(errors="replace")
            releases = VMM_RELEASE.findall(log)
            release_ok = len(releases) >= 4 and all(
                all(int(value) == 0 for value in release)
                for release in releases)
            row = {
                **case, "peak_total_mib": peak, "sampled_gpus": gpu_count,
                "m10_peak_total_mib": old_peak,
                "peak_delta_mib": None if old_peak is None else peak - old_peak,
                "peak_reduced_or_equal": int(
                    old_peak is not None and peak <= old_peak),
                "vmm_release_ok": int(release_ok),
            }
            rows.append(row)
            if gpu_count != 4 or not release_ok:
                failures.append(row)
    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "peak_total_mib", "sampled_gpus", "m10_peak_total_mib",
        "peak_delta_mib", "peak_reduced_or_equal", "vmm_release_ok",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, counts, failures


def main():
    result_root = Path(sys.argv[1])
    history_root = Path(sys.argv[2])
    rows, performance_counts = read_performance(result_root)
    comparisons, performance_failures, initial_failures, confirmed = compare_performance(
        result_root, history_root, rows)
    crossovers = write_crossovers(result_root, history_root, rows)
    memory, memory_counts, memory_failures = write_memory(
        result_root, history_root)
    smoke_counts = {
        name: len(list((result_root / name / "raw" / "smoke").glob("*.ok")))
        for name in COMPRESSORS
    }
    layout_rows = list(csv.DictReader(
        (result_root / "layout_choice.csv").open()))
    host_ok = all("PASS" in (result_root / name).read_text(errors="replace")
                  for name in ("host-size-query.txt", "host-workspace.txt"))
    layout_ok = "PASS" in (
        result_root / "layout-correctness.txt").read_text(errors="replace")
    complete = (
        performance_counts == EXPECTED_PERFORMANCE and
        smoke_counts == EXPECTED_SMOKE and
        memory_counts == EXPECTED_MEMORY and len(layout_rows) == 48 and
        len(rows) == 2 * sum(EXPECTED_PERFORMANCE.values()))
    passed = (complete and host_ok and layout_ok and
              not performance_failures and not memory_failures)
    reduced = sum(row["peak_reduced_or_equal"] for row in memory)
    gated = len(comparisons)

    with (result_root / "report.md").open("w") as stream:
        stream.write("# M11 Encoded Size Bound And B+E Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write("## Completeness\n\n")
        stream.write(f"- Performance processes: {performance_counts} / {EXPECTED_PERFORMANCE}.\n")
        stream.write(f"- Smoke processes: {smoke_counts} / {EXPECTED_SMOKE}.\n")
        stream.write(f"- Memory processes: {memory_counts} / {EXPECTED_MEMORY}.\n")
        stream.write(f"- Planner layout rows: {len(layout_rows)} / 48.\n")
        stream.write(f"- Host size/planner tests: {'PASS' if host_ok else 'FAIL'}.\n")
        stream.write(f"- GPU Pack/Unpack test: {'PASS' if layout_ok else 'FAIL'}.\n\n")
        stream.write("## Performance\n\n")
        stream.write(f"- Gated out-of-place points at >=512 MiB: {gated}.\n")
        stream.write(f"- Initial one-sided M9/M10 or historical-formula failures: {initial_failures}.\n")
        stream.write(f"- Failures resolved by independent or paired confirmation: {confirmed}.\n")
        stream.write(f"- Unresolved performance failures: {len(performance_failures)}.\n")
        stream.write("- Faster points pass; only slowdowns above the configured limit fail.\n")
        stream.write(f"- Reported crossover groups: {len(crossovers)}.\n")
        stream.write("- AllReduce OneShot stops at 32 MiB; RS TwoShot and AR TripleShot were not run.\n\n")
        stream.write("## Workspace And Memory\n\n")
        stream.write("- `layout_choice.csv` records Unified/Split selection after a single logical-shape pass.\n")
        stream.write("- Fixed-size estimators reduce common overlap from about 2B toward B+E; unavailable estimates retain E=B.\n")
        stream.write(f"- Process peaks reduced or matched M10 in {reduced}/{len(memory)} sampled cases.\n")
        stream.write(f"- Sampling or VMM release failures: {len(memory_failures)}.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
