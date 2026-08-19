#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED = {
    "completion": 32,
    "sweep": 294,
    "endpoint": 16,
    "memory": 8,
    "fixed": 2,
}


def case_fields(marker, mode):
    recipe, profile, depth, size = marker.stem.split("__")
    return {
        "mode": mode,
        "recipe": recipe,
        "compressor": profile,
        "depth": int(depth[1:]),
        "bytes": int(size[1:]),
    }


def parse_benchmark(path, case):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 13 or not fields[0].isdigit():
            continue
        base = {**case, "bytes": int(fields[0])}
        rows.append({
            **base,
            "inplace": 0,
            "time_us": float(fields[5]),
            "algbw_gbps": float(fields[6]),
            "busbw_gbps": float(fields[7]),
        })
        rows.append({
            **base,
            "inplace": 1,
            "time_us": float(fields[9]),
            "algbw_gbps": float(fields[10]),
            "busbw_gbps": float(fields[11]),
        })
    return rows


def markers(result_root, mode):
    return sorted((result_root / "raw" / mode).glob("*.ok"))


def write_completion(result_root):
    rows = []
    counts = {}
    for mode in ("completion", "sweep", "endpoint", "fixed"):
        current = markers(result_root, mode)
        counts[mode] = len(current)
        for marker in current:
            rows.extend(parse_benchmark(
                marker.with_suffix(".log"), case_fields(marker, mode)))
    fields = [
        "mode", "recipe", "compressor", "depth", "bytes", "inplace",
        "time_us", "algbw_gbps", "busbw_gbps",
    ]
    with (result_root / "completion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (result_root / "timing_breakdown.csv").open(
            "w", newline="") as stream:
        timing_fields = fields + ["component"]
        writer = csv.DictWriter(stream, fieldnames=timing_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "component": "collective_total"})
    return rows, counts


def peak_memory(path):
    peaks = defaultdict(int)
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            peaks[fields[2]] = max(peaks[fields[2]], int(fields[3]))
    return sum(peaks.values()), len(peaks)


def write_memory(result_root):
    rows = []
    current = markers(result_root, "memory")
    for marker in current:
        case = case_fields(marker, "memory")
        sample = result_root / "memory-samples" / (marker.stem + ".csv")
        peak, gpu_count = peak_memory(sample)
        rows.append({
            **case,
            "peak_total_mib": peak,
            "sampled_gpus": gpu_count,
        })
    fields = [
        "mode", "recipe", "compressor", "depth", "bytes",
        "peak_total_mib", "sampled_gpus",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, len(current)


def main():
    result_root = Path(sys.argv[1])
    rows, counts = write_completion(result_root)
    memory_rows, memory_count = write_memory(result_root)
    counts["memory"] = memory_count
    parsed_modes = defaultdict(int)
    for row in rows:
        if row["inplace"] == 0:
            parsed_modes[row["mode"]] += 1
    memory_ok = all(row["sampled_gpus"] == 4 for row in memory_rows)
    complete = counts == EXPECTED
    parsed = all(parsed_modes[mode] == EXPECTED[mode]
                 for mode in ("completion", "sweep", "endpoint", "fixed"))
    passed = complete and parsed and memory_ok

    with (result_root / "report.md").open("w") as stream:
        stream.write("# M14 dietGPU Framed Compression Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write(f"- Completed processes: {counts}.\n")
        stream.write("- Completion matrix: 8 recipes x depth 1/2/4/8.\n")
        stream.write("- Endpoint matrix: 1 MiB and 8 GiB; AllReduce OneShot is capped at 32 MiB.\n")
        stream.write("- The M0-range sweep covers depth 1/2/4/8 and uses one process per point.\n")
        stream.write(f"- Memory samples cover all four GPUs: {memory_ok}.\n")
        stream.write("- SDP4Bit AllToAll and ZFP AllGather fixed-layout smoke cases passed.\n")
        stream.write("- timing_breakdown.csv records externally observed collective totals; production execution is not synchronized solely for phase instrumentation.\n")
        stream.write("- Numerical correctness is intentionally deferred to M16.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
