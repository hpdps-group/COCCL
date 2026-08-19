#!/usr/bin/env python3

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


CORRECTNESS_RE = re.compile(r"(\w+)=([^ ]+)")
GATE_MIN_BYTES = 1 << 29
SIZES = [1 << power for power in range(20, 34)]
ORACLE_SIZES = [4 << 20, 32 << 20, 64 << 20, 512 << 20, 1 << 30, 8 << 30]
DEPTHS = [1, 2, 4, 8]


def read_status(log):
    status_path = log.with_suffix(".status")
    return int(status_path.read_text().strip()) if status_path.exists() else 0


def parse_correctness(result_root):
    rows = []
    for implementation in ("baseline", "current"):
        raw = result_root / "raw" / implementation / "hierarchical"
        for marker in sorted(raw.glob("*.ok")):
            log = marker.with_suffix(".log")
            for line in log.read_text(errors="replace").splitlines():
                if not line.startswith("COCCL_CORRECTNESS "):
                    continue
                fields = dict(CORRECTNESS_RE.findall(line))
                rows.append(
                    {
                        "implementation": implementation,
                        "operation": fields["operation"],
                        "algorithm": fields["algorithm"],
                        "compressor": fields["compressor"],
                        "datatype": fields["dtype"],
                        "depth": int(fields["depth"]),
                        "rank_count": int(fields["rank_count"]),
                        "output_elements_per_rank": int(fields["output_elements"]),
                        "mean_relative_error": float(fields["mean_relative_error"]),
                        "reference_error": "",
                        "limit": "",
                        "status": "REFERENCE" if implementation == "baseline" else "",
                    }
                )

    references = {
        (row["operation"], row["compressor"], row["datatype"]): row
        for row in rows
        if row["implementation"] == "baseline" and math.isfinite(row["mean_relative_error"])
    }
    floors = {"float": 1.0e-6, "half": 1.0e-3, "bfloat16": 1.0e-2}
    for row in rows:
        if row["implementation"] != "current":
            continue
        reference = references.get(
            (row["operation"], row["compressor"], row["datatype"])
        )
        if reference is None:
            depth_one = next(
                candidate
                for candidate in rows
                if candidate["implementation"] == "current"
                and candidate["operation"] == row["operation"]
                and candidate["compressor"] == row["compressor"]
                and candidate["datatype"] == row["datatype"]
                and candidate["depth"] == 1
            )
            reference_error = depth_one["mean_relative_error"]
            row["status"] = "NO_INITIAL_REFERENCE"
        else:
            reference_error = reference["mean_relative_error"]
        limit = max(10.0 * reference_error, floors[row["datatype"]])
        row["reference_error"] = reference_error
        row["limit"] = limit
        if not math.isfinite(row["mean_relative_error"]):
            row["status"] = "FAIL"
        elif row["mean_relative_error"] <= limit:
            if row["status"] != "NO_INITIAL_REFERENCE":
                row["status"] = "PASS"
        else:
            row["status"] = "FAIL"

    fields = [
        "implementation", "operation", "algorithm", "compressor",
        "datatype", "depth", "rank_count", "output_elements_per_rank",
        "mean_relative_error", "reference_error", "limit", "status",
    ]
    with (result_root / "correctness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    expected = 12 + 48
    failed = [row for row in rows if row["status"] == "FAIL"]
    return len(rows), expected, failed


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "phase": parts[0],
        "implementation": parts[1],
        "operation": parts[2],
        "algorithm": parts[3],
        "compressor": parts[4],
        "depth": int(parts[5][1:]),
        "bytes": int(parts[6][1:]),
    }


def parse_perf_log(log):
    case = case_fields(log)
    status = read_status(log)
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) != 13 or not tokens[0].isdigit():
            continue
        base = {
            **case,
            "bytes": int(tokens[0]),
            "exit_status": status,
            "teardown_crash": int(status != 0 and case["implementation"] == "baseline"),
            "log": str(log),
        }
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


def parse_pack(result_root):
    timings = {}
    rows = []
    for marker in sorted((result_root / "raw" / "pack").glob("*.ok")):
        log = marker.with_suffix(".log")
        for line in log.read_text(errors="replace").splitlines():
            fields = line.split(",")
            if len(fields) != 5 or not fields[0].isdigit():
                continue
            row = {
                "bytes": int(fields[0]),
                "chunks": int(fields[1]),
                "depth": int(fields[2]),
                "mode": fields[3],
                "time_us": float(fields[4]),
            }
            rows.append(row)
            timings[(row["bytes"], row["depth"], row["mode"])] = row["time_us"]
    with (result_root / "pack_unpack.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["bytes", "chunks", "depth", "mode", "time_us"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return timings, len(rows)


def pack_overhead(timings, operation, depth, size):
    overhead = timings[(size, depth, "swizzle")]
    if operation == "allreduce" and depth > 1:
        overhead += timings[(size, depth, "plain-unpack")]
    return overhead


def write_performance(result_root, pack_timings):
    rows = []
    for marker in sorted((result_root / "raw" / "performance").glob("*.ok")):
        rows.extend(parse_perf_log(marker.with_suffix(".log")))
    fields = [
        "implementation", "operation", "algorithm", "compressor", "depth",
        "bytes", "inplace", "time_us", "algbw_gbps", "busbw_gbps",
        "exit_status", "teardown_crash", "log",
    ]
    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_key = {
        (
            row["implementation"], row["operation"], row["algorithm"],
            row["compressor"], row["depth"], row["bytes"], row["inplace"],
        ): row
        for row in rows
    }
    comparisons = []
    for key, current in sorted(by_key.items()):
        if key[0] != "current":
            continue
        baseline = by_key.get(("baseline", *key[1:]))
        if baseline is None:
            continue
        overhead = 0.0
        gate = current["bytes"] >= GATE_MIN_BYTES
        if gate:
            overhead = pack_overhead(
                pack_timings, current["operation"], current["depth"], current["bytes"]
            )
        limit = baseline["time_us"] * 1.05 + overhead
        passed = current["exit_status"] == 0 and (not gate or current["time_us"] <= limit)
        comparisons.append(
            {
                "operation": current["operation"],
                "algorithm": current["algorithm"],
                "compressor": current["compressor"],
                "depth": current["depth"],
                "bytes": current["bytes"],
                "inplace": current["inplace"],
                "baseline_time_us": baseline["time_us"],
                "current_time_us": current["time_us"],
                "pack_unpack_us": overhead,
                "limit_us": limit,
                "relative_delta": (current["time_us"] - baseline["time_us"]) / baseline["time_us"],
                "baseline_teardown_crash": baseline["teardown_crash"],
                "status": "PASS" if gate and passed else ("FAIL" if gate else "RECORDED"),
            }
        )
    comparison_fields = [
        "operation", "algorithm", "compressor", "depth", "bytes", "inplace",
        "baseline_time_us", "current_time_us", "pack_unpack_us", "limit_us",
        "relative_delta", "baseline_teardown_crash", "status",
    ]
    with (result_root / "performance_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)

    native_rows = {
        (row["operation"], row["bytes"], row["inplace"]): row
        for row in rows
        if row["implementation"] == "native"
    }
    native_comparisons = []
    for row in rows:
        if row["implementation"] not in ("baseline", "current"):
            continue
        native = native_rows.get((row["operation"], row["bytes"], row["inplace"]))
        if native is None:
            continue
        native_comparisons.append(
            {
                "implementation": row["implementation"],
                "operation": row["operation"],
                "algorithm": row["algorithm"],
                "compressor": row["compressor"],
                "depth": row["depth"],
                "bytes": row["bytes"],
                "inplace": row["inplace"],
                "native_time_us": native["time_us"],
                "compressed_time_us": row["time_us"],
                "speedup": native["time_us"] / row["time_us"],
            }
        )
    native_fields = [
        "implementation", "operation", "algorithm", "compressor", "depth",
        "bytes", "inplace", "native_time_us", "compressed_time_us", "speedup",
    ]
    with (result_root / "native_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=native_fields)
        writer.writeheader()
        writer.writerows(native_comparisons)

    expected_rows = (2 * 2 * 2 * 4 * len(SIZES) + 2 * len(SIZES)) * 2
    failed = [row for row in comparisons if row["status"] == "FAIL"]
    return rows, expected_rows, comparisons, failed


def write_oracle(result_root, performance_rows):
    candidate_rows = [
        row for row in performance_rows
        if row["implementation"] == "current" and row["inplace"] == 0
    ]
    for marker in sorted((result_root / "raw" / "oracle").glob("*.ok")):
        candidate_rows.extend(
            row for row in parse_perf_log(marker.with_suffix(".log"))
            if row["inplace"] == 0
        )
    wanted = []
    for compressor in ("sdp4bit", "zfp"):
        for depth in (1, 4):
            for size in ORACLE_SIZES:
                wanted.extend(
                    [
                        ("reducescatter", "oneshot", compressor, depth, size),
                        ("reducescatter", "twoshot", compressor, depth, size),
                        ("allreduce", "twoshot", compressor, depth, size),
                        ("allreduce", "tripleshot", compressor, depth, size),
                    ]
                )
                if size <= 32 << 20:
                    wanted.append(("allreduce", "oneshot", compressor, depth, size))
    by_candidate = {
        (row["operation"], row["algorithm"], row["compressor"], row["depth"], row["bytes"]): row
        for row in candidate_rows
    }
    selected = [by_candidate[key] for key in wanted if key in by_candidate]
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["operation"], row["compressor"], row["depth"], row["bytes"])].append(row)
    output = []
    for key, candidates in sorted(grouped.items()):
        best = min(row["time_us"] for row in candidates)
        best_set = sorted(
            row["algorithm"] for row in candidates if row["time_us"] <= best * 1.03
        )
        for row in sorted(candidates, key=lambda item: item["algorithm"]):
            output.append(
                {
                    "topology": "2x4",
                    "rank_count": 8,
                    "operation": key[0],
                    "compressor": key[1],
                    "datatype": "float",
                    "depth": key[2],
                    "bytes": key[3],
                    "candidate": row["algorithm"],
                    "elapsed_time_us": row["time_us"],
                    "measured_best_time_us": best,
                    "measured_best_set": ";".join(best_set),
                    "within_best_set": int(row["algorithm"] in best_set),
                    "log": row["log"],
                }
            )
    fields = [
        "topology", "rank_count", "operation", "compressor", "datatype",
        "depth", "bytes", "candidate", "elapsed_time_us",
        "measured_best_time_us", "measured_best_set", "within_best_set", "log",
    ]
    with (result_root / "candidate_oracle.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return len(output), len(wanted)


def main():
    result_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/data/home/scyb672/run/lxc/COCCL-migrate/results/M17"
    )
    correctness_rows, expected_correctness, correctness_failed = parse_correctness(result_root)
    pack_timings, pack_rows = parse_pack(result_root)
    performance_rows, expected_performance, comparisons, performance_failed = (
        write_performance(result_root, pack_timings)
    )
    oracle_rows, expected_oracle = write_oracle(result_root, performance_rows)
    failed_current_processes = [
        row for row in performance_rows
        if row["implementation"] == "current" and row["exit_status"] != 0
    ]
    passed = (
        correctness_rows == expected_correctness
        and not correctness_failed
        and pack_rows == 60
        and len(performance_rows) == expected_performance
        and not performance_failed
        and not failed_current_processes
        and oracle_rows == expected_oracle
    )
    max_excess = max(
        (
            (row["current_time_us"] - row["limit_us"]) / row["limit_us"]
            for row in comparisons
            if row["status"] == "FAIL"
        ),
        default=0.0,
    )
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M17 Hierarchical Algorithms Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write("- Topology: 2 nodes x 4 A800 GPUs (8 ranks).\n")
        stream.write(f"- Correctness rows: {correctness_rows}/{expected_correctness}.\n")
        stream.write(f"- Correctness failures: {len(correctness_failed)}.\n")
        stream.write(f"- Pack/Unpack timing rows: {pack_rows}/60.\n")
        stream.write(f"- Performance rows: {len(performance_rows)}/{expected_performance}.\n")
        stream.write("- Native NCCL rows cover both collectives and every message size.\n")
        stream.write(f"- Large-message performance failures: {len(performance_failed)}.\n")
        stream.write(f"- Maximum limit excess: {max_excess * 100.0:.3f}%.\n")
        stream.write(f"- Current nonzero exits with timing: {len(failed_current_processes)}.\n")
        stream.write(f"- Candidate oracle rows: {oracle_rows}/{expected_oracle}.\n")
        stream.write(
            "- SDP4Bit hierarchical comparison uses Hadamard disabled on both "
            "implementations; the initial repository config enables it, while "
            "the migrated fused DRC path does not support it.\n"
        )
        stream.write("- Initial COCCL depth 2/4/8 is performance-only.\n")
        stream.write("- Autotune remained disabled; M18 consumes the held-out oracle.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
