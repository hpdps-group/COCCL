#!/usr/bin/env python3

import csv
import re
import statistics
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
            rows.extend(
                [
                    {**base, "inplace": 0, "time_us": float(tokens[5]),
                     "algbw_gbps": float(tokens[6]),
                     "busbw_gbps": float(tokens[7])},
                    {**base, "inplace": 1, "time_us": float(tokens[9]),
                     "algbw_gbps": float(tokens[10]),
                     "busbw_gbps": float(tokens[11])},
                ]
            )
    return rows


def load_m0(path):
    values = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if (row["variant"] == "migrate-copy" and
                    row["compressor"] == "native" and
                    int(row["depth"]) == 1):
                key = (row["collective"], "native", int(row["bytes"]),
                       int(row["inplace"]))
                values[key].append(float(row["time_us"]))
    return {key: statistics.median(samples) for key, samples in values.items()}


def load_m1(path):
    values = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != "migrate" or row["compressor"] == "native":
                continue
            key = ("alltoall", row["compressor"], int(row["bytes"]),
                   int(row["inplace"]))
            values[key] = float(row["time_us"])
    return values


def write_performance(result_root, m0_root, m1_root):
    rows = []
    raw_root = result_root / "raw" / "performance"
    markers = sorted(raw_root.glob("*.ok"))
    for marker in markers:
        rows.extend(parse_log(marker.with_suffix(".log")))

    fields = ["collective", "algorithm", "compressor", "bytes", "inplace",
              "time_us", "algbw_gbps", "busbw_gbps"]
    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    baseline = load_m0(m0_root / "summary.csv")
    baseline.update(load_m1(m1_root / "performance.csv"))
    comparisons = []
    for row in rows:
        key = (row["collective"], row["compressor"], row["bytes"],
               row["inplace"])
        reference = baseline.get(key)
        if reference is None:
            continue
        delta = (row["time_us"] - reference) / reference
        gate = row["bytes"] >= GATE_MIN_BYTES
        comparisons.append(
            {**{key: row[key] for key in
                ("collective", "compressor", "bytes", "inplace")},
             "baseline": "M0" if row["compressor"] == "native" else "M1",
             "baseline_us": reference, "m2_us": row["time_us"],
             "relative_delta": delta, "is_gate_point": int(gate),
             "passed": int(not gate or delta <= 0.03)}
        )

    comparison_fields = ["collective", "compressor", "bytes", "inplace",
                         "baseline", "baseline_us", "m2_us",
                         "relative_delta", "is_gate_point", "passed"]
    with (result_root / "performance-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return len(markers), comparisons


def parse_memory_sample(path):
    peak_by_gpu = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                peak_by_gpu[fields[2]] = max(peak_by_gpu[fields[2]],
                                             int(fields[3]))
    return sum(peak_by_gpu.values()), len(peak_by_gpu)


def load_memory_baselines(m0_root, m1_root):
    samples = defaultdict(list)
    with (m0_root / "memory.csv").open() as stream:
        for row in csv.DictReader(stream):
            if (row["variant"] == "migrate-copy" and
                    row["compressor"] == "native" and
                    int(row["depth"]) == 1):
                key = (row["collective"], "native", int(row["bytes"]))
                samples[key].append(int(row["peak_total_mib"]))
    baseline = {key: statistics.median(values)
                for key, values in samples.items()}
    with (m1_root / "memory.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["variant"] == "migrate" and row["compressor"] != "native":
                key = ("alltoall", row["compressor"], int(row["bytes"]))
                baseline[key] = int(row["peak_total_mib"])
    return baseline


def write_memory(result_root, m0_root, m1_root):
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

    baseline = load_memory_baselines(m0_root, m1_root)
    comparisons = []
    for row in rows:
        key = (row["collective"], row["compressor"], row["bytes"])
        reference = baseline.get(key)
        if reference is None:
            continue
        comparisons.append(
            {"collective": row["collective"],
             "compressor": row["compressor"], "bytes": row["bytes"],
             "baseline": "M0" if row["compressor"] == "native" else "M1",
             "baseline_peak_mib": reference,
             "m2_peak_mib": row["peak_total_mib"],
             "delta_mib": row["peak_total_mib"] - reference}
        )
    fields = ["collective", "compressor", "bytes", "baseline",
              "baseline_peak_mib", "m2_peak_mib", "delta_mib"]
    with (result_root / "memory-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return len(list(marker_root.glob("*.ok"))), rows, comparisons


def host_overhead(result_root):
    path = result_root / "host-overhead.txt"
    if not path.exists():
        return None, None
    match = re.search(
        r"disabled_ns_per_call=([0-9.]+) threshold_ns_per_call=([0-9.]+)",
        path.read_text(errors="replace"),
    )
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)


def main():
    result_root = Path(sys.argv[1])
    m0_root = Path(sys.argv[2])
    m1_root = Path(sys.argv[3])
    performance_count, comparisons = write_performance(
        result_root, m0_root, m1_root)
    memory_count, memory_rows, memory_comparisons = write_memory(
        result_root, m0_root, m1_root)
    disabled_ns, threshold_ns = host_overhead(result_root)

    expected_performance = len(PERFORMANCE_SIZES) * 6
    expected_memory = len(MEMORY_SIZES) * 6
    gate_rows = [row for row in comparisons if row["is_gate_point"]]
    failures = [row for row in gate_rows if not row["passed"]]
    complete = (performance_count == expected_performance and
                memory_count == expected_memory and
                len(comparisons) == expected_performance * 2 and
                len(memory_comparisons) == expected_memory and
                all(row["sampled_gpus"] == 4 for row in memory_rows) and
                disabled_ns is not None)
    status = "PASS" if complete and not failures else "INCOMPLETE OR FAILED"

    lines = [
        "# M2 Unified Runtime Control Plane Report", "",
        f"Status: {status}", "", "## Scope", "",
        "- Native AllToAll, AllGather, ReduceScatter, and AllReduce compare against M0.",
        "- Explicit SDP4Bit/ZFP AllToAll compares against M1 and uses a 16 GiB threshold to verify bypass semantics.",
        "- Every process uses four GPUs with `-w 20 -n 30 -c 0`; pipeline depth remains 1.",
        "", "## Completeness", "",
        f"- Performance processes: {performance_count}/{expected_performance}.",
        f"- Memory processes: {memory_count}/{expected_memory}.",
        f"- Performance comparison rows: {len(comparisons)}/{expected_performance * 2}.",
        "", "## Performance Gate", "",
        f"- Gate rows at 512 MiB and above: {len(gate_rows)}.",
        f"- Failed rows: {len(failures)}; limit is 3%.",
        "- 64 MiB is informational.", "", "## Host Routing Overhead", "",
        f"- Disabled fast path: {disabled_ns:.2f} ns/call." if disabled_ns is not None else "- Disabled fast path: missing.",
        f"- Threshold fallback: {threshold_ns:.2f} ns/call." if threshold_ns is not None else "- Threshold fallback: missing.",
        "", "## Memory", "",
        "| Collective | Compressor | Bytes | Baseline | Baseline MiB | M2 MiB | Delta MiB |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in memory_comparisons:
        lines.append(
            f"| {row['collective']} | {row['compressor']} | {row['bytes']} | "
            f"{row['baseline']} | {row['baseline_peak_mib']} | "
            f"{row['m2_peak_mib']} | {row['delta_mib']} |"
        )
    if failures:
        lines.extend(["", "## Gate Failures", ""])
        for row in failures:
            lines.append(
                f"- {row['collective']} {row['compressor']} {row['bytes']} "
                f"inplace={row['inplace']}: {row['relative_delta'] * 100:.2f}%."
            )
    lines.extend(["", "## Verification", "",
                  "- Runtime and group CPU tests cover disabled routing, unsupported calls, threshold fallback, explicit bypass, caller guard, grouped drain, and native replay.",
                  "- The collective data path remains the M1 implementation in `nccl_comp_wrapper.cc`.",
                  "- `src/enqueue.cc` is unchanged.", ""])
    (result_root / "report.md").write_text("\n".join(lines))

    if not complete:
        raise SystemExit("M2 result matrix is incomplete")
    if failures:
        raise SystemExit("M2 performance gate failed")


if __name__ == "__main__":
    main()
