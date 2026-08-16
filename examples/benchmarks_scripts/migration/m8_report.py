#!/usr/bin/env python3

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GATE_BYTES = 512 << 20
EXPECTED_SMOKE = 37
EXPECTED_PERFORMANCE = 162
EXPECTED_LAYOUT = 56
EXPECTED_MEMORY = 26


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
            rows.append({
                **base,
                "inplace": 0,
                "time_us": float(tokens[5]),
                "algbw_gbps": float(tokens[6]),
                "busbw_gbps": float(tokens[7]),
            })
            rows.append({
                **base,
                "inplace": 1,
                "time_us": float(tokens[9]),
                "algbw_gbps": float(tokens[10]),
                "busbw_gbps": float(tokens[11]),
            })
    return rows


def write_allreduce(result_root):
    raw = result_root / "raw" / "performance"
    rows = []
    markers = sorted(raw.glob("allreduce__*.ok"))
    for marker in markers:
        rows.extend(parse_benchmark(marker.with_suffix(".log"),
                                    case_fields(marker)))
    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "time_us", "algbw_gbps", "busbw_gbps",
    ]
    with (result_root / "allreduce.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, len(markers)


def write_layout(result_root):
    raw = result_root / "raw" / "layout"
    rows = []
    markers = sorted(raw.glob("layout__*.ok"))
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
    return rows, len(markers)


def baseline_medians(baseline_root):
    grouped = defaultdict(list)
    with (baseline_root / "summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != "original" or row["collective"] != "allreduce":
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
        baseline_compressor = (
            "cuzfp" if row["compressor"] == "zfp" else row["compressor"]
        )
        key = (
            row["algorithm"], baseline_compressor, row["depth"],
            row["bytes"], row["inplace"],
        )
        old_us = baseline.get(key)
        if old_us is None:
            continue
        layout_us = layout.get((row["bytes"], row["depth"]), 0.0)
        gate = (
            row["bytes"] >= GATE_BYTES
            and row["inplace"] == 0
            and row["algorithm"] != "oneshot"
        )
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
            **{name: row[name] for name in (
                "algorithm", "compressor", "depth", "bytes", "inplace")},
            "m0_us": old_us,
            "m8_us": row["time_us"],
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
        "m0_us", "m8_us", "layout_us", "limit_us", "relative_delta",
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
            if len(fields) == 4:
                peaks[fields[2]] = max(peaks[fields[2]], int(fields[3]))
    return sum(peaks.values()), len(peaks)


def workspace_plans(result_root):
    grouped = defaultdict(list)
    with (result_root / "workspace-plan.csv").open() as stream:
        for row in csv.DictReader(stream):
            grouped[(row["algorithm"], int(row["bytes"]),
                     int(row["requested_depth"]))].append({
                "temp_index": int(row["temp_index"]),
                "temp_role": row["temp_role"],
                "logical_bytes": int(row["logical_bytes"]),
                "aligned_bytes": int(row["aligned_bytes"]),
                "offset": int(row["offset"]),
                "slice_workspace_bytes": int(row["slice_workspace_bytes"]),
                "workspace_bytes": int(row["workspace_bytes"]),
            })
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
    markers = sorted([
        *raw.glob("allreduce__*.ok"),
        *raw.glob("allreduce__*.oom"),
    ])
    oom_cases = []
    for marker in markers:
        case = case_fields(marker)
        status = "oom" if marker.suffix == ".oom" else "ok"
        if status == "oom":
            oom_cases.append((case["algorithm"], case["compressor"],
                              case["depth"], case["bytes"]))
        peak, gpu_count = peak_memory(samples / (marker.stem + ".csv"))
        release_ok = ""
        if status == "ok":
            releases = RELEASE.findall(
                marker.with_suffix(".log").read_text(errors="replace")
            )
            release_ok = int(len(releases) >= 4 and all(
                all(int(value) == 0 for value in release)
                for release in releases
            ))
            if not release_ok:
                release_failures.append(marker.stem)
        key = (case["algorithm"], case["bytes"], case["depth"])
        for plan in plans[key]:
            rows.append({
                "algorithm": case["algorithm"],
                "compressor": case["compressor"],
                "depth": case["depth"],
                "bytes": case["bytes"],
                "status": status,
                "peak_total_mib": peak,
                "sampled_gpus": gpu_count,
                "release_ok": release_ok,
                **plan,
            })
    fields = [
        "algorithm", "compressor", "depth", "bytes", "status",
        "peak_total_mib", "sampled_gpus", "release_ok", "temp_index", "temp_role",
        "logical_bytes", "aligned_bytes", "offset",
        "slice_workspace_bytes", "workspace_bytes",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, len(markers), oom_cases, release_failures


def main():
    result_root = Path(sys.argv[1])
    baseline_root = Path(sys.argv[2])
    allreduce_rows, performance_processes = write_allreduce(result_root)
    layout_rows, layout_processes = write_layout(result_root)
    comparisons, gate_failures = compare_performance(
        result_root, allreduce_rows, layout_rows, baseline_root
    )
    memory_rows, memory_processes, memory_oom, release_failures = write_memory(
        result_root
    )
    smoke_processes = len(list((result_root / "raw" / "smoke").glob("*.ok")))
    host_ok = all(
        "PASS" in (result_root / name).read_text(errors="replace")
        for name in ("host-plan.txt", "host-recipe.txt",
                     "layout-correctness.txt")
    )
    memory_gpu_ok = bool(memory_rows) and all(
        row["sampled_gpus"] == 4 for row in memory_rows
    )
    unexpected_memory_oom = memory_oom
    complete = (
        smoke_processes == EXPECTED_SMOKE
        and performance_processes == EXPECTED_PERFORMANCE
        and len(allreduce_rows) == EXPECTED_PERFORMANCE * 2
        and layout_processes == EXPECTED_LAYOUT
        and len(layout_rows) == EXPECTED_LAYOUT
        and memory_processes == EXPECTED_MEMORY
    )
    passed = (
        complete and host_ok and memory_gpu_ok
        and not gate_failures and not release_failures
        and not unexpected_memory_oom
    )
    status = "PASS" if passed else "FAIL"

    gated = [row for row in comparisons if row["gate"]]
    small = [row for row in comparisons
             if row["bytes"] < GATE_BYTES and row["inplace"] == 0]
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M8 Pipeline AllReduce Report\n\n")
        stream.write(f"Status: {status}\n\n")
        stream.write("## Scope\n\n")
        stream.write("- OneShot: Compress -> AllGather -> DecompressReduce.\n")
        stream.write("- TwoShot: Compress -> AllToAll -> DRC -> AllGather -> Decompress.\n")
        stream.write("- TripleShot is compile/Host-only for this single-node milestone.\n")
        stream.write("- SDP4Bit and ZFP use `-w 20 -n 30 -c 0`, one process per point.\n")
        stream.write("- OneShot runs from 4 KiB through 32 MiB; its memory sample is 32 MiB.\n\n")
        stream.write("## Completeness\n\n")
        stream.write(f"- Smoke processes: {smoke_processes}/{EXPECTED_SMOKE}.\n")
        stream.write(f"- Performance processes: {performance_processes}/{EXPECTED_PERFORMANCE}; rows: {len(allreduce_rows)}/{EXPECTED_PERFORMANCE * 2}.\n")
        stream.write(f"- Layout processes/rows: {layout_processes}/{EXPECTED_LAYOUT}, {len(layout_rows)}/{EXPECTED_LAYOUT}.\n")
        stream.write(f"- Memory processes: {memory_processes}/{EXPECTED_MEMORY}; temp rows: {len(memory_rows)}.\n")
        stream.write(f"- Memory capacity limits: {len(memory_oom)}; unexpected: {len(unexpected_memory_oom)}.\n")
        stream.write(f"- Host/layout checks: {'PASS' if host_ok else 'FAIL'}.\n\n")
        stream.write("## Performance Gate\n\n")
        stream.write(f"- Gated out-of-place native/TwoShot points at >=512 MiB: {len(gated)}.\n")
        stream.write(f"- Gate failures: {len(gate_failures)}.\n")
        stream.write(f"- Crossover-only out-of-place points: {len(small)} plus all OneShot points.\n")
        stream.write("- `T_layout` is Pack + Unpack from the matching M8 layout run.\n\n")
        stream.write("## Workspace And Lifetime\n\n")
        stream.write("- Planner records OneShot and TwoShot temp capacity before Arena reuse.\n")
        stream.write(f"- Memory samples cover four GPUs: {'PASS' if memory_gpu_ok else 'FAIL'}.\n")
        stream.write(f"- VMM teardown failures: {len(release_failures)}.\n")
        stream.write("- Detailed logical/aligned temp sizes and sampled peaks are in `memory.csv`.\n\n")
        stream.write("## Notes\n\n")
        stream.write("- Non-divisible AllReduce counts remain on native NCCL before a compressed plan is built.\n")
        stream.write("- In-place columns are informational; overlapping buffers use the serial fallback.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
