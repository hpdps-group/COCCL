#!/usr/bin/env python3

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GATE_BYTES = 512 << 20
EXPECTED_SMOKE = 9
EXPECTED_PERFORMANCE = 126
EXPECTED_LAYOUT = 56
EXPECTED_MEMORY = 24


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0],
        "algorithm": parts[1],
        "compressor": parts[2],
        "depth": int(parts[3][1:]),
        "bytes": int(parts[4][1:]),
    }


def parse_benchmark(path, case):
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            base = {**case, "bytes": int(tokens[0])}
            rows.append(
                {
                    **base,
                    "inplace": 0,
                    "time_us": float(tokens[5]),
                    "algbw_gbps": float(tokens[6]),
                    "busbw_gbps": float(tokens[7]),
                }
            )
            rows.append(
                {
                    **base,
                    "inplace": 1,
                    "time_us": float(tokens[9]),
                    "algbw_gbps": float(tokens[10]),
                    "busbw_gbps": float(tokens[11]),
                }
            )
    return rows


def write_alltoall(result_root):
    raw = result_root / "raw" / "performance"
    rows = []
    for marker in sorted(raw.glob("alltoall__*.ok")):
        rows.extend(parse_benchmark(marker.with_suffix(".log"),
                                    case_fields(marker)))
    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "time_us", "algbw_gbps", "busbw_gbps",
    ]
    with (result_root / "alltoall.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, len(list(raw.glob("alltoall__*.ok")))


def write_layout(result_root):
    raw = result_root / "raw" / "layout"
    rows = []
    for marker in sorted(raw.glob("layout__*.ok")):
        with marker.with_suffix(".log").open(errors="replace") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                rows.append(
                    {
                        "bytes": int(row["bytes"]),
                        "chunks": int(row["chunks"]),
                        "depth": int(row["depth"]),
                        "pack_us": float(row["pack_us"]),
                        "unpack_us": float(row["unpack_us"]),
                        "layout_us": float(row["layout_us"]),
                    }
                )
    fields = ["bytes", "chunks", "depth", "pack_us", "unpack_us",
              "layout_us"]
    with (result_root / "layout.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, len(list(raw.glob("layout__*.ok")))


def baseline_medians(baseline_root):
    grouped = defaultdict(list)
    with (baseline_root / "summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != "original" or row["collective"] != "alltoall":
                continue
            key = (
                row["algorithm"], row["compressor"], int(row["depth"]),
                int(row["bytes"]), int(row["inplace"]),
            )
            grouped[key].append(float(row["time_us"]))
    return {key: statistics.median(values) for key, values in grouped.items()}


def compare_performance(result_root, rows, layout_rows, baseline_root):
    baseline = baseline_medians(baseline_root)
    layout = {(row["bytes"], row["depth"]): row["layout_us"]
              for row in layout_rows}
    comparisons = []
    failures = []
    for row in rows:
        baseline_compressor = "cuzfp" if row["compressor"] == "zfp" else row["compressor"]
        key = (
            row["algorithm"], baseline_compressor, row["depth"],
            row["bytes"], row["inplace"],
        )
        old_us = baseline.get(key)
        if old_us is None:
            continue
        layout_us = layout.get((row["bytes"], row["depth"]), 0.0)
        gate = row["bytes"] >= GATE_BYTES and row["inplace"] == 0
        if row["algorithm"] == "native":
            limit_us = old_us * 1.03
            rule = "native+3%"
        elif row["depth"] == 1:
            limit_us = old_us * 1.05
            rule = "depth1+5%"
        else:
            limit_us = old_us * 1.05 + layout_us
            rule = "baseline+layout+5%"
        passed = not gate or row["time_us"] <= limit_us
        comparison = {
            **{key: row[key] for key in (
                "algorithm", "compressor", "depth", "bytes", "inplace")},
            "m0_us": old_us,
            "m5_us": row["time_us"],
            "layout_us": layout_us,
            "limit_us": limit_us,
            "relative_delta": (row["time_us"] - old_us) / old_us,
            "gate": int(gate),
            "rule": rule,
            "passed": int(passed),
        }
        comparisons.append(comparison)
        if not passed:
            failures.append(comparison)

    fields = [
        "algorithm", "compressor", "depth", "bytes", "inplace",
        "m0_us", "m5_us", "layout_us", "limit_us", "relative_delta",
        "gate", "rule", "passed",
    ]
    with (result_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons, failures


def peak_memory(path):
    peaks = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            peaks[fields[2]] = max(peaks[fields[2]], int(fields[3]))
    return sum(peaks.values()), len(peaks)


def workspace_plans(result_root):
    grouped = defaultdict(list)
    with (result_root / "workspace-plan.csv").open() as stream:
        for row in csv.DictReader(stream):
            converted = {
                "temp_index": int(row["temp_index"]),
                "temp_role": row["temp_role"],
                "logical_bytes": int(row["logical_bytes"]),
                "aligned_bytes": int(row["aligned_bytes"]),
                "offset": int(row["offset"]),
                "slice_workspace_bytes": int(row["slice_workspace_bytes"]),
                "workspace_bytes": int(row["workspace_bytes"]),
            }
            grouped[(int(row["bytes"]), int(row["requested_depth"]))].append(converted)
    return grouped


RELEASE = re.compile(
    r"COCCL VMM release .*remaining_virtual (\d+) "
    r"remaining_physical (\d+) remaining_registered (\d+)"
)


def write_memory(result_root):
    raw = result_root / "raw" / "memory"
    samples = result_root / "memory-samples"
    plans = workspace_plans(result_root)
    rows = []
    release_failures = []
    marker_count = 0
    for marker in sorted(raw.glob("alltoall__*.ok")):
        marker_count += 1
        case = case_fields(marker)
        sample = samples / (marker.stem + ".csv")
        peak, gpu_count = peak_memory(sample)
        releases = RELEASE.findall(marker.with_suffix(".log").read_text(errors="replace"))
        release_ok = len(releases) >= 4 and all(
            all(int(value) == 0 for value in release) for release in releases
        )
        if not release_ok:
            release_failures.append(marker.stem)
        for plan in plans[(case["bytes"], case["depth"])]:
            rows.append(
                {
                    "compressor": case["compressor"],
                    "depth": case["depth"],
                    "bytes": case["bytes"],
                    "peak_total_mib": peak,
                    "sampled_gpus": gpu_count,
                    "release_ok": int(release_ok),
                    **plan,
                }
            )
    fields = [
        "compressor", "depth", "bytes", "peak_total_mib", "sampled_gpus",
        "release_ok", "temp_index", "temp_role", "logical_bytes",
        "aligned_bytes", "offset", "slice_workspace_bytes", "workspace_bytes",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, marker_count, release_failures


def main():
    result_root = Path(sys.argv[1])
    baseline_root = Path(sys.argv[2])
    alltoall_rows, performance_processes = write_alltoall(result_root)
    layout_rows, layout_processes = write_layout(result_root)
    comparisons, gate_failures = compare_performance(
        result_root, alltoall_rows, layout_rows, baseline_root
    )
    memory_rows, memory_processes, release_failures = write_memory(result_root)
    smoke_processes = len(list((result_root / "raw" / "smoke").glob("*.ok")))
    host_ok = all(
        "PASS" in (result_root / name).read_text(errors="replace")
        for name in ("host-plan.txt", "host-stage.txt", "layout-correctness.txt")
    )
    memory_gpu_ok = all(row["sampled_gpus"] == 4 for row in memory_rows)
    complete = (
        smoke_processes == EXPECTED_SMOKE
        and performance_processes == EXPECTED_PERFORMANCE
        and len(alltoall_rows) == EXPECTED_PERFORMANCE * 2
        and layout_processes == EXPECTED_LAYOUT
        and len(layout_rows) == EXPECTED_LAYOUT
        and memory_processes == EXPECTED_MEMORY
    )
    passed = complete and host_ok and memory_gpu_ok and not gate_failures and not release_failures

    gated = [row for row in comparisons if row["gate"]]
    hard_deltas = [row["relative_delta"] for row in gated]
    small = [row for row in comparisons if row["bytes"] < GATE_BYTES and row["inplace"] == 0]
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M5 Pipeline AllToAll Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write("## Scope\n\n")
        stream.write("- First Pipeline path: Pack -> Compress -> AllToAll -> Decompress -> Unpack.\n")
        stream.write("- SDP4Bit and ZFP, depth 1/2/4/8; native fallback measured separately.\n")
        stream.write("- Performance uses `-w 20 -n 30 -c 0`, one process per point.\n\n")
        stream.write("## Completeness\n\n")
        stream.write(f"- Smoke processes: {smoke_processes}/{EXPECTED_SMOKE}.\n")
        stream.write(f"- Performance processes: {performance_processes}/{EXPECTED_PERFORMANCE}; rows: {len(alltoall_rows)}/{EXPECTED_PERFORMANCE * 2}.\n")
        stream.write(f"- Layout processes/rows: {layout_processes}/{EXPECTED_LAYOUT}, {len(layout_rows)}/{EXPECTED_LAYOUT}.\n")
        stream.write(f"- Memory processes: {memory_processes}/{EXPECTED_MEMORY}; temp rows: {len(memory_rows)}.\n")
        stream.write(f"- Host/layout correctness checks: {'PASS' if host_ok else 'FAIL'}.\n\n")
        stream.write("## Performance Gate\n\n")
        stream.write(f"- Gated out-of-place points at >=512 MiB: {len(gated)}.\n")
        stream.write(f"- Gate failures: {len(gate_failures)}.\n")
        if hard_deltas:
            stream.write(f"- Largest gated regression before layout allowance: {max(hard_deltas) * 100:.3f}%.\n")
        stream.write(f"- Sub-512 MiB out-of-place points reported only: {len(small)}.\n\n")
        stream.write("## Workspace And Lifetime\n\n")
        stream.write("- Depth 1 plans two independent raw-capacity temps; depth 2/4/8 plan four independent per-slice temps.\n")
        stream.write(f"- Memory samples cover four GPUs: {'PASS' if memory_gpu_ok else 'FAIL'}.\n")
        stream.write(f"- VMM teardown failures: {len(release_failures)}.\n")
        stream.write("- Detailed logical/aligned temp sizes and sampled peaks are in `memory.csv`.\n\n")
        stream.write("## Notes\n\n")
        stream.write("- AllToAll numerical correctness remains outside this migration gate; benchmark completion and guard checks are recorded.\n")
        stream.write("- In-place columns are informational because the canonical AllToAll test does not claim in-place support; overlapping buffers fall back to depth 1.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
