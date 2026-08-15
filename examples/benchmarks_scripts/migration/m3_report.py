#!/usr/bin/env python3

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


PERFORMANCE_SIZES = (1 << 26, 1 << 29, 1 << 30, 1 << 33)
MEMORY_SIZES = (1 << 26, 1 << 30, 1 << 33)
GATE_MIN_BYTES = 1 << 29


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0],
        "algorithm": parts[1],
        "compressor": parts[2],
        "bytes": int(parts[3][1:]),
    }


def parse_log(path):
    case = case_fields(path)
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            base = {**case, "bytes": int(tokens[0])}
            rows.extend([
                {**base, "inplace": 0, "time_us": float(tokens[5]),
                 "algbw_gbps": float(tokens[6]),
                 "busbw_gbps": float(tokens[7])},
                {**base, "inplace": 1, "time_us": float(tokens[9]),
                 "algbw_gbps": float(tokens[10]),
                 "busbw_gbps": float(tokens[11])},
            ])
    return rows


def load_performance(path):
    values = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != "alltoall":
                continue
            key = (row["compressor"], int(row["bytes"]),
                   int(row["inplace"]))
            values[key] = float(row["time_us"])
    return values


def write_performance(result_root, m2_root):
    rows = []
    marker_root = result_root / "raw" / "performance"
    markers = sorted(marker_root.glob("*.ok"))
    for marker in markers:
        rows.extend(parse_log(marker.with_suffix(".log")))

    fields = ["collective", "algorithm", "compressor", "bytes", "inplace",
              "time_us", "algbw_gbps", "busbw_gbps"]
    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    baseline = load_performance(m2_root / "performance.csv")
    comparisons = []
    for row in rows:
        key = (row["compressor"], row["bytes"], row["inplace"])
        reference = baseline.get(key)
        if reference is None:
            continue
        delta = (row["time_us"] - reference) / reference
        gate = row["bytes"] >= GATE_MIN_BYTES
        comparisons.append({
            "compressor": row["compressor"],
            "bytes": row["bytes"],
            "inplace": row["inplace"],
            "m2_us": reference,
            "m3_us": row["time_us"],
            "relative_delta": delta,
            "is_gate_point": int(gate),
            "passed": int(not gate or delta <= 0.03),
        })

    fields = ["compressor", "bytes", "inplace", "m2_us", "m3_us",
              "relative_delta", "is_gate_point", "passed"]
    with (result_root / "performance-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return len(markers), rows, comparisons


def parse_memory_sample(path):
    peak_by_gpu = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                peak_by_gpu[fields[2]] = max(peak_by_gpu[fields[2]],
                                             int(fields[3]))
    return sum(peak_by_gpu.values()), len(peak_by_gpu)


def load_memory(path):
    values = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != "alltoall":
                continue
            key = (row["compressor"], int(row["bytes"]))
            values[key] = int(row["peak_total_mib"])
    return values


def write_memory(result_root, m2_root):
    rows = []
    marker_root = result_root / "raw" / "memory"
    for sample in sorted((result_root / "memory-samples").glob("*.csv")):
        if not (marker_root / (sample.stem + ".ok")).exists():
            continue
        peak, gpus = parse_memory_sample(sample)
        rows.append({**case_fields(sample), "peak_total_mib": peak,
                     "sampled_gpus": gpus})

    fields = ["collective", "algorithm", "compressor", "bytes",
              "peak_total_mib", "sampled_gpus"]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    baseline = load_memory(m2_root / "memory.csv")
    comparisons = []
    for row in rows:
        reference = baseline.get((row["compressor"], row["bytes"]))
        if reference is None:
            continue
        comparisons.append({
            "compressor": row["compressor"],
            "bytes": row["bytes"],
            "m2_peak_mib": reference,
            "m3_peak_mib": row["peak_total_mib"],
            "delta_mib": row["peak_total_mib"] - reference,
        })

    fields = ["compressor", "bytes", "m2_peak_mib", "m3_peak_mib",
              "delta_mib"]
    with (result_root / "memory-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return len(list(marker_root.glob("*.ok"))), rows, comparisons


def write_lifecycle(result_root):
    allocation_pattern = re.compile(
        r"COCCL legacy buffer comm (0x[0-9a-fA-F]+) allocated "
        r"([0-9]+) bytes, total ([0-9]+)")
    release_pattern = re.compile(
        r"COCCL legacy buffer comm (0x[0-9a-fA-F]+) released "
        r"([0-9]+) bytes")
    allocations = defaultdict(list)
    releases = {}
    lifecycle_root = result_root / "raw" / "lifecycle"
    markers = sorted(lifecycle_root.glob("*.ok"))
    for marker in markers:
        with marker.with_suffix(".log").open(errors="replace") as stream:
            for line in stream:
                match = allocation_pattern.search(line)
                if match:
                    allocations[match.group(1)].append(
                        (int(match.group(2)), int(match.group(3))))
                match = release_pattern.search(line)
                if match:
                    releases[match.group(1)] = int(match.group(2))

    rows = []
    for comm, events in sorted(allocations.items()):
        allocated = sum(event[0] for event in events)
        released = releases.get(comm, -1)
        rows.append({
            "comm": comm,
            "allocation_count": len(events),
            "allocated_bytes": allocated,
            "released_bytes": released,
            "stable_high_water": int(len(events) <= 2),
            "released_all": int(released == allocated),
        })

    fields = ["comm", "allocation_count", "allocated_bytes",
              "released_bytes", "stable_high_water", "released_all"]
    with (result_root / "lifecycle.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(markers), rows


def main():
    result_root = Path(sys.argv[1])
    m2_root = Path(sys.argv[2])
    performance_count, performance_rows, comparisons = write_performance(
        result_root, m2_root)
    memory_count, memory_rows, memory_comparisons = write_memory(
        result_root, m2_root)
    lifecycle_count, lifecycle_rows = write_lifecycle(result_root)

    expected_performance = len(PERFORMANCE_SIZES) * 3
    expected_memory = len(MEMORY_SIZES) * 3
    gate_rows = [row for row in comparisons if row["is_gate_point"]]
    failures = [row for row in gate_rows if not row["passed"]]
    lifecycle_passed = (
        len(lifecycle_rows) == 4 and
        all(row["stable_high_water"] and row["released_all"]
            for row in lifecycle_rows)
    )
    complete = (
        performance_count == expected_performance and
        len(performance_rows) == expected_performance * 2 and
        len(comparisons) == expected_performance * 2 and
        memory_count == expected_memory and
        len(memory_rows) == expected_memory and
        len(memory_comparisons) == expected_memory and
        all(row["sampled_gpus"] == 4 for row in memory_rows) and
        lifecycle_count == 1 and lifecycle_passed
    )
    status = "PASS" if complete and not failures else "INCOMPLETE OR FAILED"

    lines = [
        "# M3 Communicator-bound Buffer Manager Report", "",
        f"Status: {status}", "", "## Scope", "",
        "- Migrated the active M2 compressed collective workspace and generic reduction temporary storage to communicator-owned legacy pools.",
        "- Workspace byte formulas, collective recipes, compressor plugins, and pipeline depth remain unchanged.",
        "- Pending slices are reused without synchronization on the same CUDA stream; different streams remain isolated until the completion event is ready.",
        "- VMM and Pipeline/Arena planning remain deferred to later migration stages.",
        "", "## Completeness", "",
        f"- Performance processes: {performance_count}/{expected_performance}.",
        f"- Performance rows: {len(performance_rows)}/{expected_performance * 2}.",
        f"- Memory processes: {memory_count}/{expected_memory}.",
        f"- Lifecycle processes: {lifecycle_count}/1.",
        "", "## Performance Gate", "",
        f"- Gate rows at 512 MiB and above: {len(gate_rows)}.",
        f"- Failed rows: {len(failures)}; maximum regression is 3% against M2.",
        "- 64 MiB is informational.",
        "", "## Memory", "",
        "| Compressor | Bytes | M2 MiB | M3 MiB | Delta MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in memory_comparisons:
        lines.append(
            f"| {row['compressor']} | {row['bytes']} | "
            f"{row['m2_peak_mib']} | {row['m3_peak_mib']} | "
            f"{row['delta_mib']} |"
        )
    lines.extend([
        "", "## Lifecycle", "",
        "| Comm | Allocations | Allocated bytes | Released bytes | Stable | Released all |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in lifecycle_rows:
        lines.append(
            f"| {row['comm']} | {row['allocation_count']} | "
            f"{row['allocated_bytes']} | {row['released_bytes']} | "
            f"{row['stable_high_water']} | {row['released_all']} |"
        )
    lines.extend([
        "", "## Verification", "",
        "- Host tests cover in-use exclusion, reuse after release, independent communicator pools, registration rollback, release retry, and teardown with zero live allocations/events/registrations.",
        "- The lifecycle run uses 20 warmups and 30 iterations; allocation events plateau while communicator teardown releases the recorded high-water bytes.",
        "- Memory samples use one independent process per case and four GPUs.",
    ])
    if failures:
        lines.extend(["", "## Gate Failures", ""])
        for row in failures:
            lines.append(
                f"- {row['compressor']} {row['bytes']} inplace={row['inplace']}: "
                f"{row['relative_delta'] * 100:.2f}%."
            )
    (result_root / "report.md").write_text("\n".join(lines) + "\n")

    if not complete:
        raise SystemExit("M3 result matrix or lifecycle evidence is incomplete")
    if failures:
        raise SystemExit("M3 performance gate failed")


if __name__ == "__main__":
    main()
