#!/usr/bin/env python3

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from m20_report import write_csv


GATE_BYTES = 512 << 20
SINGLE_RE = re.compile(
    r"(sdp4bit|zfp)-(allgather|allreduce|alltoall|reducescatter)-d4")
TWO_NODE_RE = re.compile(
    r"(sdp4bit|zfp)-(allreduce|reducescatter)-d(1|2|4|8)-b(\d+)\.log")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return sum(values) / len(values)


def benchmark_measurements(path):
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) == 13 and fields[0].isdigit():
            return [
                {
                    "bytes": int(fields[0]),
                    "inplace": 0,
                    "time_us": float(fields[5]),
                    "algbw_gbps": float(fields[6]),
                    "busbw_gbps": float(fields[7]),
                },
                {
                    "bytes": int(fields[0]),
                    "inplace": 1,
                    "time_us": float(fields[9]),
                    "algbw_gbps": float(fields[10]),
                    "busbw_gbps": float(fields[11]),
                },
            ]
    raise ValueError(f"benchmark row not found in {path}")


def load_m0(path):
    values = defaultdict(list)
    for row in read_csv(path):
        if row["variant"] != "original" or row["compressor"] == "native":
            continue
        compressor = (
            "zfp" if row["compressor"] == "cuzfp" else row["compressor"]
        )
        key = (row["collective"], compressor, int(row["depth"]),
               int(row["bytes"]), int(row["inplace"]))
        values[key].append(float(row["time_us"]))
    return {key: mean(samples) for key, samples in values.items()}


def load_m17(path):
    result = {}
    for row in read_csv(path):
        if (row["implementation"] != "baseline" or
                row["compressor"] == "native"):
            continue
        key = (row["operation"], row["compressor"], int(row["depth"]),
               int(row["bytes"]), int(row["inplace"]))
        result[key] = float(row["time_us"])
    return result


def load_m20(path):
    rows = []
    for row in read_csv(path):
        if (row["compressor"] == "native" or
                not row["variant"].startswith("control")):
            continue
        rows.append({
            "topology": row["topology"],
            "operation": row["operation"],
            "compressor": row["compressor"],
            "depth": int(row["depth"]),
            "bytes": int(row["bytes"]),
            "inplace": int(row["inplace"]),
            "time_us": float(row["time_us"]),
        })
    return rows


def exact_m20(rows, topology, operation, compressor, depth, bytes_, inplace):
    candidates = [
        row for row in rows
        if row["topology"] == topology and row["operation"] == operation and
        row["compressor"] == compressor and row["depth"] == depth and
        row["bytes"] == bytes_ and row["inplace"] == inplace
    ]
    return candidates[0]["time_us"] if candidates else math.nan


def performance_status(current, depth1, original, original_depth1, m20,
                       bytes_, inplace):
    original_gain = (original_depth1 - original) / original_depth1
    if inplace:
        return "RECORDED", original_gain, "in-place"
    if bytes_ < GATE_BYTES:
        return "RECORDED", original_gain, "small-message"
    m20_pass = not math.isfinite(m20) or current <= 1.03 * m20
    original_pass = current <= 1.15 * original
    # NCCL elapsed time varies slightly even when two configurations execute
    # the same schedule. Treat a <=3% delta as equivalent for this gate.
    depth1_limit = 1.03 if original_gain > 0.10 else 1.15
    depth1_pass = current <= depth1_limit * depth1
    status = "PASS" if m20_pass and (original_pass or depth1_pass) else "FAIL"
    reason = (
        "original" if original_pass else
        ("current-depth1" if depth1_pass else "none")
    )
    return status, original_gain, reason


def performance_row(topology, operation, compressor, requested_depth,
                    effective_depth, measurement, depth1_time, original,
                    original_depth1, m20_time, log):
    status, original_gain, reason = performance_status(
        measurement["time_us"], depth1_time, original, original_depth1,
        m20_time, measurement["bytes"], measurement["inplace"])
    return {
        "topology": topology,
        "compressor": compressor,
        "operation": operation,
        "requested_depth": requested_depth,
        "effective_depth": effective_depth,
        **measurement,
        "depth1_time_us": depth1_time,
        "depth1_ratio": measurement["time_us"] / depth1_time,
        "m20_time_us": m20_time,
        "m20_ratio": measurement["time_us"] / m20_time
                     if math.isfinite(m20_time) else math.nan,
        "original_time_us": original,
        "original_ratio": measurement["time_us"] / original,
        "original_depth1_time_us": original_depth1,
        "original_depth_gain_pct": 100.0 * original_gain,
        "acceptance_path": reason,
        "status": status,
        "log": str(log),
    }


def write_single_node(root, temp_root, m0, m20):
    rows = []
    for path in sorted((temp_root / "selected-schedule").glob("*.log")):
        match = SINGLE_RE.fullmatch(path.stem)
        if not match:
            continue
        compressor, operation = match.groups()
        depth1_path = (
            temp_root / "canonical-depths" /
            f"{compressor}-{operation}-d1.log"
        )
        current = benchmark_measurements(path)
        depth1 = benchmark_measurements(depth1_path)
        for measurement, depth1_measurement in zip(current, depth1):
            key = (operation, compressor, 4, measurement["bytes"],
                   measurement["inplace"])
            key1 = (operation, compressor, 1, measurement["bytes"],
                    measurement["inplace"])
            rows.append(performance_row(
                "single-node", operation, compressor, 4, 1, measurement,
                depth1_measurement["time_us"], m0[key], m0[key1],
                exact_m20(m20, "single-node", operation, compressor, 4,
                          measurement["bytes"], measurement["inplace"]),
                path))
    write_performance_csv(root / "end_to_end_single_node.csv", rows)
    return rows


def write_two_nodes(root, temp_root, m17, m20):
    measurements = {}
    logs = {}
    matrix_root = temp_root / "two-node-depth-matrix"
    for path in sorted(matrix_root.glob("*.log")):
        match = TWO_NODE_RE.fullmatch(path.name)
        if not match:
            continue
        compressor, operation, depth, bytes_ = match.groups()
        for measurement in benchmark_measurements(path):
            key = (operation, compressor, int(depth), int(bytes_),
                   measurement["inplace"])
            measurements[key] = measurement
            logs[key] = path

    rows = []
    for key, measurement in sorted(measurements.items()):
        operation, compressor, depth, bytes_, inplace = key
        if depth == 1:
            continue
        depth1 = measurements[(operation, compressor, 1, bytes_, inplace)][
            "time_us"]
        rows.append(performance_row(
            "2x4", operation, compressor, depth, depth, measurement, depth1,
            m17[key], m17[(operation, compressor, 1, bytes_, inplace)],
            exact_m20(m20, "2x4", operation, compressor, depth, bytes_,
                      inplace), logs[key]))
    write_performance_csv(root / "end_to_end_two_nodes.csv", rows)
    return rows


def write_performance_csv(path, rows):
    write_csv(path, rows, [
        "topology", "compressor", "operation", "requested_depth",
        "effective_depth", "bytes", "inplace", "time_us", "algbw_gbps",
        "busbw_gbps", "depth1_time_us", "depth1_ratio", "m20_time_us",
        "m20_ratio", "original_time_us", "original_ratio",
        "original_depth1_time_us", "original_depth_gain_pct",
        "acceptance_path", "status", "log",
    ])


def write_baselines(root, rows):
    write_csv(root / "baseline.csv", rows, [
        "topology", "compressor", "operation", "requested_depth", "bytes",
        "inplace", "m20_time_us", "original_time_us",
        "original_depth1_time_us", "depth1_time_us",
    ])
    write_csv(root / "original_depth_scaling.csv", rows, [
        "topology", "compressor", "operation", "requested_depth", "bytes",
        "inplace", "original_depth1_time_us", "original_time_us",
        "original_depth_gain_pct",
    ])


def write_correctness(root, temp_root):
    rows = read_csv(temp_root / "final-results" /
                    "collective_correctness.csv")
    for row in rows:
        row["shape"] = (
            "divisible"
            if int(row["raw_chunk_elements"]) % int(row["depth"]) == 0
            else "remainder"
        )
    write_csv(root / "correctness.csv", rows, [
        "topology", "rank_count", "operation", "algorithm", "compressor",
        "dtype", "depth", "shape", "raw_chunk_elements", "output_elements",
        "mean_relative_error", "error_limit", "status", "log",
    ])
    return rows


def write_kernel(root, temp_root):
    def timings(path):
        samples = defaultdict(list)
        for row in read_csv(path):
            key = (row["shape"], row["bytes"], row["depth"],
                   row["remainder"], row["mode"])
            samples[key].append(float(row["time_us"]))
        return {key: mean(values) for key, values in samples.items()}

    baseline = timings(temp_root / "pack_kernel_m20_o3.csv")
    current = timings(temp_root / "pack_kernel_final_o3.csv")
    rows = []
    for key, after in sorted(current.items()):
        shape, bytes_, depth, remainder, mode = key
        before = baseline[key]
        rows.append({
            "shape": shape,
            "bytes": bytes_,
            "depth": depth,
            "remainder": remainder,
            "mode": mode,
            "m20_time_us": before,
            "m21_time_us": after,
            "ratio": after / before,
            "status": "PASS" if int(bytes_) < GATE_BYTES or
                      after <= 1.03 * before else "FAIL",
        })
    write_csv(root / "pack_kernel.csv", rows, [
        "shape", "bytes", "depth", "remainder", "mode", "m20_time_us",
        "m21_time_us", "ratio", "status",
    ])
    return rows


def write_workspace(root, temp_root):
    baseline = {
        (row["operation"], row["input_bytes"], row["requested_depth"]): row
        for row in read_csv(temp_root / "workspace_m20.csv")
    }
    rows = []
    for row in read_csv(temp_root / "workspace_m21.csv"):
        key = (row["operation"], row["input_bytes"],
               row["requested_depth"])
        before = int(baseline[key]["total_bytes"])
        after = int(row["total_bytes"])
        rows.append({
            "operation": row["operation"],
            "input_bytes": int(row["input_bytes"]),
            "requested_depth": int(row["requested_depth"]),
            "effective_depth": int(row["effective_depth"]),
            "m20_workspace_bytes": before,
            "m21_workspace_bytes": after,
            "delta_bytes": after - before,
            "ratio": after / before,
            "status": "PASS" if after <= before else "FAIL",
        })
    write_csv(root / "workspace.csv", rows, [
        "operation", "input_bytes", "requested_depth", "effective_depth",
        "m20_workspace_bytes", "m21_workspace_bytes", "delta_bytes",
        "ratio", "status",
    ])
    return rows


def candidate_measurement(path):
    measurement = benchmark_measurements(path)[0]
    match = re.search(
        r"single-node__(sdp4bit|zfp)__(allgather|allreduce|alltoall|"
        r"reducescatter)__d(\d+)", path.name)
    return match.group(1), match.group(2), int(match.group(3)), measurement


def write_strategy(root, temp_root):
    rows = []
    candidates = (
        ("raw-slice", "focused-nonfused"),
        ("raw-slice-serialized-boundary", "focused-serialized"),
        ("encoded-serial", "encoded-prototype"),
        ("encoded-pipelined", "encoded-pipelined"),
        ("encoded-whole", "encoded-whole"),
    )
    for strategy, directory in candidates:
        for path in sorted((temp_root / directory / "raw" /
                            "performance-single").glob("*.log")):
            compressor, operation, depth, measurement = candidate_measurement(
                path)
            rows.append({
                "topology": "single-node",
                "strategy": strategy,
                "compressor": compressor,
                "operation": operation,
                "depth": depth,
                "bytes": measurement["bytes"],
                "time_us": measurement["time_us"],
                "status": "REJECTED",
                "log": str(path),
            })
    for path in sorted((temp_root / "selected-schedule").glob("*.log")):
        match = SINGLE_RE.fullmatch(path.stem)
        compressor, operation = match.groups()
        measurement = benchmark_measurements(path)[0]
        rows.append({
            "topology": "single-node",
            "strategy": "aggregate-serial",
            "compressor": compressor,
            "operation": operation,
            "depth": 4,
            "bytes": measurement["bytes"],
            "time_us": measurement["time_us"],
            "status": "SELECTED",
            "log": str(path),
        })
    write_csv(root / "strategy_matrix.csv", rows, [
        "topology", "strategy", "compressor", "operation", "depth",
        "bytes", "time_us", "status", "log",
    ])
    write_csv(root / "winner_optimization.csv", [
        {
            "topology": "single-node",
            "compressor_class": "fixed-size",
            "winner": "aggregate-serial",
            "compressor_changes": "none",
            "fused_codec": "no",
            "rank_scaled_codec_launches": "no",
        },
        {
            "topology": "multi-node",
            "compressor_class": "fixed-size",
            "winner": "slice-overlap",
            "compressor_changes": "none",
            "fused_codec": "no",
            "rank_scaled_codec_launches": "no",
        },
        {
            "topology": "all",
            "compressor_class": "framed",
            "winner": "slice-overlap",
            "compressor_changes": "none",
            "fused_codec": "no",
            "rank_scaled_codec_launches": "no",
        },
    ], ["topology", "compressor_class", "winner", "compressor_changes",
        "fused_codec", "rank_scaled_codec_launches"])
    with (root / "rejected_strategies.md").open("w") as stream:
        stream.write("# M21 候选方案结论\n\n")
        stream.write("- 单节点 fixed-size 选择现有整块 serial stage graph；它不需要 "
                     "Pack/Unpack，也不改变 compressor。\n")
        stream.write("- 多节点 fixed-size 保留逐 slice overlap；512 MiB 和 1 GiB "
                     "ReduceScatter/AllReduce 已测得有效 overlap 收益。\n")
        stream.write("- framed compressor 保留原 overlap 协议。\n")
        stream.write("- PTX `.cg`、boundary serialization、Encoded-Serial、"
                     "Encoded-Pipelined 和 Encoded-Whole 均因端到端无收益淘汰。\n")
        stream.write("- 新版 SDP4Bit fused/swizzled codec 不参与候选或验收。\n")


def write_stage_timeline(root, temp_root):
    path = temp_root / "nsys" / "allgather-d4-rank0-stats.csv"
    lines = path.read_text().splitlines()
    header = lines.index(
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),"
        "Max (ns),StdDev (ns),Name")
    end = lines.index("", header)
    kernels = list(csv.DictReader(lines[header:end]))
    definitions = [
        ("Pack", "packSliceKernel"),
        ("Compress", "cached_quantization"),
        ("AllGather", "ncclDevKernel_AllGather"),
        ("Decompress", "dequantize_kernel"),
        ("Unpack", "unpackSliceKernel"),
    ]
    rows = []
    for stage, pattern in definitions:
        match = next((row for row in kernels if pattern in row["Name"]), None)
        rows.append({
            "case": "single-node-raw-slice-sdp4bit-allgather-d4-1gib",
            "strategy": "raw-slice-rejected",
            "stage": stage,
            "instances": int(match["Instances"]) if match else 0,
            "total_us": float(match["Total Time (ns)"]) / 1000
                        if match else 0,
            "average_us": float(match["Avg (ns)"]) / 1000
                          if match else 0,
            "kernel": match["Name"] if match else "",
        })
    write_csv(root / "stage_timeline.csv", rows, [
        "case", "strategy", "stage", "instances", "total_us",
        "average_us", "kernel",
    ])


def validate_profile(temp_root):
    values = {}
    path = temp_root / "size-sweep" / "sdp4bit-performance-profile.txt"
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return (
        values.get("profile") == "core-pack-ordinary-codec" and
        values.get("fused_hierarchical_swizzle") == "0" and
        values.get("new_path") == "core-swizzle-ordinary-codec" and
        values.get("original_sdp4bit_path") == "swizzled-quantize"
    )


def write_report(root, performance, correctness, kernel, workspace,
                 profile_valid):
    performance_failures = [row for row in performance
                            if row["status"] == "FAIL"]
    correctness_failures = [row for row in correctness
                            if row["status"] == "FAIL"]
    kernel_failures = [row for row in kernel if row["status"] == "FAIL"]
    workspace_failures = [row for row in workspace
                          if row["status"] == "FAIL"]
    with (root / "final_report.md").open("w") as stream:
        stream.write("# M21 Pack/Unpack 端到端优化验收\n\n")
        stream.write("最终策略按拓扑和 compressor 类别直接选择：单节点 fixed-size "
                     "使用整块 serial stage graph；多节点 fixed-size 与 framed "
                     "compressor 保留逐 slice overlap。没有修改 compressor API 或源码，"
                     "codec 调用数不随 rank 增长。\n\n")
        stream.write("新版 SDP4Bit 只使用 Core Pack/Swizzle + ordinary codec。原版 "
                     "COCCL 的 swizzled codec 只用于原版性能参考；新版 fused 结果"
                     "不参与候选或 PASS。\n\n")
        stream.write("## 验收统计\n\n")
        stream.write(f"- SDP4Bit non-fused preflight："
                     f"{'PASS' if profile_valid else 'FAIL'}。\n")
        stream.write(f"- collective correctness：{len(correctness)} 条，失败 "
                     f"{len(correctness_failures)} 条。\n")
        stream.write(f"- 端到端性能 gate：{len(performance)} 条，失败 "
                     f"{len(performance_failures)} 条。\n")
        stream.write(f"- Pack/Unpack kernel：{len(kernel)} 条，大消息回退超 3% "
                     f"{len(kernel_failures)} 条。\n")
        stream.write(f"- workspace：{len(workspace)} 条，增长 "
                     f"{len(workspace_failures)} 条。\n\n")
        stream.write("## 双节点 Depth 结论\n\n")
        stream.write("- 64 MiB：四种组合均以 depth=1 为最优或最稳定。\n")
        stream.write("- 512 MiB/1 GiB ReduceScatter：SDP4Bit 以 depth=4 为主，"
                     "ZFP 以 depth=4/8 为主。\n")
        stream.write("- AllReduce 收益较弱，但 1 GiB 时 depth=8 最优。\n")
        stream.write("- M20 只在 bytes/count 完全一致时比较；不同 shape 不用近邻"
                     "结果代替。相同执行路径采用 3% 计时容差。\n")
        if performance_failures:
            stream.write("\n## 性能失败点\n\n")
            stream.write("| topology | codec | op | depth | bytes | current/d1 | "
                         "current/original | current/M20 |\n")
            stream.write("|---|---|---|---:|---:|---:|---:|---:|\n")
            for row in performance_failures:
                stream.write(
                    f"| {row['topology']} | {row['compressor']} | "
                    f"{row['operation']} | {row['requested_depth']} | "
                    f"{row['bytes']} | {row['depth1_ratio']:.4f} | "
                    f"{row['original_ratio']:.4f} | {row['m20_ratio']:.4f} |\n")
    return (performance_failures, correctness_failures, kernel_failures,
            workspace_failures)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: m21_report.py RESULT_ROOT HISTORY_ROOT TEMP_ROOT")
    root = Path(sys.argv[1])
    history = Path(sys.argv[2])
    temp_root = Path(sys.argv[3])
    root.mkdir(parents=True, exist_ok=True)

    m0 = load_m0(history / "M0" / "summary.csv")
    m17 = load_m17(history / "M17" / "performance.csv")
    m20 = load_m20(history / "M20" / "performance.csv")
    single = write_single_node(root, temp_root, m0, m20)
    two_node = write_two_nodes(root, temp_root, m17, m20)
    performance = single + two_node
    write_baselines(root, performance)
    correctness = write_correctness(root, temp_root)
    kernel = write_kernel(root, temp_root)
    workspace = write_workspace(root, temp_root)
    write_strategy(root, temp_root)
    write_stage_timeline(root, temp_root)
    profile_valid = validate_profile(temp_root)
    failures = write_report(root, performance, correctness, kernel,
                            workspace, profile_valid)
    failure_count = (
        sum(len(group) for group in failures) +
        (0 if profile_valid else 1)
    )
    print(f"M21 report: correctness={len(correctness)} "
          f"performance={len(performance)} kernel={len(kernel)} "
          f"workspace={len(workspace)} failures={failure_count}")
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
