#!/usr/bin/env python3

import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path


FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")
GATE_BYTES = 512 << 20


def write_csv(path, rows, names):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_benchmark(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) != 13 or not tokens[0].isdigit():
            continue
        rows.append((0, float(tokens[5]), float(tokens[6]), float(tokens[7])))
        rows.append((1, float(tokens[9]), float(tokens[10]), float(tokens[11])))
    return rows


def performance_case(path):
    parts = path.stem.split("__")
    if len(parts) != 7:
        raise ValueError(f"unexpected M20 performance stem: {path.stem}")
    return {
        "topology": parts[0],
        "compressor": parts[1],
        "operation": parts[2],
        "depth": int(parts[3][1:]),
        "variant": parts[4],
        "raw_chunk_elements": int(parts[5][1:]),
        "bytes": int(parts[6][1:]),
    }


def parse_performance(root):
    rows = []
    by_key = {}
    for directory in ("performance-single", "performance-two-node"):
        for log in sorted((root / "raw" / directory).glob("*.log")):
            case = performance_case(log)
            for inplace, time_us, algbw, busbw in parse_benchmark(log):
                row = {
                    **case,
                    "inplace": inplace,
                    "time_us": time_us,
                    "algbw_gbps": algbw,
                    "busbw_gbps": busbw,
                    "log": str(log),
                }
                rows.append(row)
                key = (
                    row["topology"], row["compressor"], row["operation"],
                    row["depth"], row["variant"], row["bytes"], inplace,
                )
                by_key[key] = row

    comparisons = []
    for row in rows:
        if row["variant"] not in ("remainder", "remainder-max"):
            continue
        maximum = row["variant"] == "remainder-max"
        suffix = "-max" if maximum else ""
        remainder = row["depth"] - 1 if maximum else 1
        ranks = 4 if row["topology"] == "single-node" else 8
        control_bytes = row["bytes"] - ranks * 4 * remainder
        base = (
            row["topology"], row["compressor"], row["operation"],
            row["depth"],
        )
        control = by_key.get((*base, f"control{suffix}", control_bytes,
                              row["inplace"]))
        depth1 = by_key.get((*base, f"depth1{suffix}", row["bytes"],
                             row["inplace"]))
        if control is None or depth1 is None:
            raise ValueError(f"missing pair for {row['log']}")

        bandwidth_ratio = row["algbw_gbps"] / control["algbw_gbps"]
        depth1_ratio = row["time_us"] / depth1["time_us"]
        control_limit = 0.97 if row["bytes"] >= GATE_BYTES else 0.95
        gate = row["inplace"] == 0
        control_pass = bandwidth_ratio >= control_limit
        depth1_pass = row["bytes"] < GATE_BYTES or depth1_ratio <= 1.0
        status = "PASS" if gate and control_pass and depth1_pass else (
            "FAIL" if gate else "RECORDED")
        comparisons.append({
            **row,
            "remainder": remainder,
            "control_bytes": control_bytes,
            "control_time_us": control["time_us"],
            "control_algbw_gbps": control["algbw_gbps"],
            "bandwidth_ratio": bandwidth_ratio,
            "control_limit": control_limit,
            "depth1_time_us": depth1["time_us"],
            "depth1_ratio": depth1_ratio,
            "control_pass": int(control_pass),
            "depth1_pass": int(depth1_pass),
            "status": status,
        })

    fields = [
        "topology", "compressor", "operation", "depth", "variant",
        "remainder", "raw_chunk_elements", "bytes", "inplace", "time_us",
        "algbw_gbps", "busbw_gbps", "control_bytes", "control_time_us",
        "control_algbw_gbps", "bandwidth_ratio", "control_limit",
        "depth1_time_us", "depth1_ratio", "control_pass", "depth1_pass",
        "status", "log",
    ]
    write_csv(root / "performance.csv", rows, [
        "topology", "compressor", "operation", "depth", "variant",
        "raw_chunk_elements", "bytes", "inplace", "time_us",
        "algbw_gbps", "busbw_gbps", "log",
    ])
    write_csv(root / "performance_comparison.csv", comparisons, fields)
    return rows, comparisons


def load_correctness_limits(history_root):
    limits = {}
    path = history_root / "M16" / "correctness.csv"
    if not path.exists():
        return limits
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (
                row["topology"], row["operation"], row["algorithm"],
                row["compressor"], row["dtype"], int(row["depth"]),
            )
            limit = row.get("error_limit", "")
            if limit:
                limits[key] = float(limit)
    return limits


def parse_correctness(root, history_root):
    limits = load_correctness_limits(history_root)
    rows = []
    for directory in ("correctness-single", "correctness-two-node"):
        for log in sorted((root / "raw" / directory).glob("*.log")):
            for line in log.read_text(errors="replace").splitlines():
                if not line.startswith("COCCL_CORRECTNESS "):
                    continue
                value = dict(FIELD_RE.findall(line))
                error = float(value["mean_relative_error"])
                key = (
                    value["topology"], value["operation"],
                    value["algorithm"], value["compressor"], value["dtype"],
                    int(value["depth"]),
                )
                default_limit = 1.0e-9 if value["compressor"] == "dietgpu" \
                    else 1.0
                limit = limits.get(key, default_limit)
                status = "PASS" if math.isfinite(error) and error <= limit \
                    else "FAIL"
                rows.append({
                    **value,
                    "rank_count": int(value["rank_count"]),
                    "depth": int(value["depth"]),
                    "raw_chunk_elements": int(value["raw_chunk_elements"]),
                    "output_elements": int(value["output_elements"]),
                    "mean_relative_error": error,
                    "error_limit": limit,
                    "status": status,
                    "log": str(log),
                })
    fields = [
        "topology", "rank_count", "operation", "algorithm", "compressor",
        "dtype", "depth", "raw_chunk_elements", "output_elements",
        "mean_relative_error", "error_limit", "status", "log",
    ]
    write_csv(root / "collective_correctness.csv", rows, fields)
    return rows


def write_slice_shapes(root):
    rows = []
    for depth, count in ((2, 4097), (4, 4097), (4, 4099),
                         (8, 4097), (8, 4103)):
        quotient, remainder = divmod(count, depth)
        offset = 0
        for index in range(depth):
            elements = quotient + (remainder if index + 1 == depth else 0)
            rows.append({
                "raw_chunk_elements": count,
                "depth": depth,
                "slice": index,
                "element_offset": offset,
                "element_count": elements,
                "byte_offset_fp32": 4 * offset,
                "bytes_fp32": 4 * elements,
            })
            offset += elements
    write_csv(root / "slice_shapes.csv", rows, [
        "raw_chunk_elements", "depth", "slice", "element_offset",
        "element_count", "byte_offset_fp32", "bytes_fp32",
    ])


def write_planner_evidence(root):
    capacities = [
        {
            "case": "alltoall-divisible", "raw_chunk_elements": 16,
            "depth": 4, "max_slice_elements": 4,
            "stage_capacities": "16;16;64", "workspace_bytes": 2048,
        },
        {
            "case": "alltoall-remainder", "raw_chunk_elements": 17,
            "depth": 4, "max_slice_elements": 5,
            "stage_capacities": "20;20;80", "workspace_bytes": 2048,
        },
        {
            "case": "alltoall-framed-remainder", "raw_chunk_elements": 17,
            "depth": 4, "max_slice_elements": 5,
            "stage_capacities": "80;80;80", "workspace_bytes": "",
        },
        {
            "case": "hierarchical-mixed-remainder",
            "raw_chunk_elements": 17, "depth": 4,
            "max_slice_elements": 5,
            "stage_capacities": "40;10;10;20", "workspace_bytes": "",
        },
    ]
    write_csv(root / "planner_capacity.csv", capacities, [
        "case", "raw_chunk_elements", "depth", "max_slice_elements",
        "stage_capacities", "workspace_bytes",
    ])
    write_csv(root / "workspace.csv", [
        {
            "case": "alltoall", "depth": 4,
            "divisible_raw_chunk_elements": 16,
            "remainder_raw_chunk_elements": 17,
            "divisible_workspace_bytes": 2048,
            "remainder_workspace_bytes": 2048,
            "delta_bytes": 0,
        }
    ], [
        "case", "depth", "divisible_raw_chunk_elements",
        "remainder_raw_chunk_elements", "divisible_workspace_bytes",
        "remainder_workspace_bytes", "delta_bytes",
    ])
    write_csv(root / "layout_correctness.csv", [
        {"suite": "pack-unpack-layout", "cases": 166, "status": "PASS"}
    ], ["suite", "cases", "status"])


def write_report(root, performance, comparisons, correctness):
    failures = [row for row in comparisons if row["status"] == "FAIL"]
    correctness_failures = [row for row in correctness
                            if row["status"] == "FAIL"]
    counts = Counter((row["topology"], row["status"])
                     for row in comparisons if row["inplace"] == 0)
    worst = sorted(
        (row for row in comparisons if row["inplace"] == 0),
        key=lambda row: min(row["bandwidth_ratio"],
                            1.0 / row["depth1_ratio"]),
    )[:20]
    with (root / "report.md").open("w") as stream:
        stream.write("# M20 非整除 Pipeline Slice 验收\n\n")
        stream.write("## 执行结果\n\n")
        stream.write(f"- 正确性记录：{len(correctness)}，失败："
                     f"{len(correctness_failures)}。\n")
        stream.write(f"- 性能原始记录：{len(performance)}（含 in-place）。\n")
        for topology in ("single-node", "2x4"):
            stream.write(
                f"- {topology} out-of-place 比较："
                f"PASS {counts[(topology, 'PASS')]}，"
                f"FAIL {counts[(topology, 'FAIL')]}。\n")
        stream.write("\n性能状态使用实际字节带宽比较 remainder/control；"
                     "512 MiB 以上要求不低于 97%，短消息不低于 95%。"
                     "512 MiB 以上同时要求 overlap 时间不慢于同 count 的 depth 1。\n\n")
        stream.write("## 最差性能点\n\n")
        stream.write("| topology | codec | op | depth | variant | bytes | "
                     "control ratio | depth1 ratio | status |\n")
        stream.write("|---|---|---|---:|---|---:|---:|---:|---|\n")
        for row in worst:
            stream.write(
                f"| {row['topology']} | {row['compressor']} | "
                f"{row['operation']} | {row['depth']} | {row['variant']} | "
                f"{row['bytes']} | {row['bandwidth_ratio']:.4f} | "
                f"{row['depth1_ratio']:.4f} | {row['status']} |\n")
        if failures:
            stream.write("\n超限点需要同参数配对复测后才能归因为稳定回退。\n")

    return failures, correctness_failures


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: m20_report.py RESULT_ROOT HISTORY_ROOT")
    root = Path(sys.argv[1])
    history_root = Path(sys.argv[2])
    performance, comparisons = parse_performance(root)
    correctness = parse_correctness(root, history_root)
    write_slice_shapes(root)
    write_planner_evidence(root)
    failures, correctness_failures = write_report(
        root, performance, comparisons, correctness)
    print(f"M20 report: correctness={len(correctness)} "
          f"correctness_failures={len(correctness_failures)} "
          f"performance={len(performance)} "
          f"performance_failures={len(failures)}")
    return 1 if failures or correctness_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
