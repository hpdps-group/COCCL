#!/usr/bin/env python3

import csv
import math
import re
import sys
from pathlib import Path


FIELDS_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def fields(line):
    return dict(FIELDS_RE.findall(line))


def write_csv(path, rows, names):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(stream, headers, rows):
    stream.write("| " + " | ".join(headers) + " |\n")
    stream.write("|" + "|".join("---" for _ in headers) + "|\n")
    for row in rows:
        stream.write("| " + " | ".join(str(value) for value in row) + " |\n")


SCOPES = {
    "default-sdp": ("sdp4bit:4/128", "inherit-default", "inherit-default"),
    "default-zfp": ("zfp:8", "inherit-default", "inherit-default"),
    "intra-sdp": ("disabled", "sdp4bit:4/128", "disabled"),
    "inter-sdp": ("disabled", "disabled", "sdp4bit:4/128"),
    "split-sdp": ("disabled", "sdp4bit:8/1024", "sdp4bit:4/512"),
    "default-intra-override": (
        "sdp4bit:4/128", "sdp4bit:8/1024", "inherit-default"),
    "default-inter-override": (
        "sdp4bit:8/1024", "inherit-default", "sdp4bit:4/512"),
    "disable-intra": ("sdp4bit:8/1024", "disabled", "sdp4bit:4/512"),
    "disable-inter": ("sdp4bit:4/128", "sdp4bit:8/1024", "disabled"),
    "mixed-sdp-zfp": ("disabled", "sdp4bit:8/1024", "zfp:8"),
    "dietgpu-inter": ("disabled", "disabled", "dietgpu:10"),
}

EXPECTED_CORRECTNESS_ROWS = 10 * 2 * 3 * 2
EXPECTED_PERFORMANCE_ROWS = 2 * 3 * 2 + 2 * 5 * 2 * 3 * 2
EXPECTED_DIETGPU_CORRECTNESS_ROWS = 2 * 2 + 1
EXPECTED_DIETGPU_PERFORMANCE_ROWS = 2 * 2 * 2 * 3 + 2 * 3


def resolve(entry, default):
    return default if entry == "inherit-default" else entry


def drc_mode(decoder, encoder):
    return "fused" if (decoder == encoder and
                       decoder.startswith("sdp4bit:")) else "generic"


def write_scope_tables(root):
    config_rows = []
    effective_rows = []
    for name, (default, intra, inter) in SCOPES.items():
        config_rows.append({
            "case": name, "default": default, "intra": intra, "inter": inter,
            "inter_has_input_parameters": False,
        })
        effective_rows.append({
            "case": name,
            "default": default,
            "intra": resolve(intra, default),
            "inter": resolve(inter, default),
        })
    write_csv(root / "config_cases.csv", config_rows,
              ["case", "default", "intra", "inter",
               "inter_has_input_parameters"])
    write_csv(root / "effective_policies.csv", effective_rows,
              ["case", "default", "intra", "inter"])
    return effective_rows


def graph_for(case_name, operation):
    default, intra, inter = SCOPES[case_name]
    intra_codec = resolve(intra, default)
    inter_codec = resolve(inter, default)
    intra_enabled = intra_codec != "disabled"
    inter_enabled = inter_codec != "disabled"
    final_enabled = default != "disabled"
    stages = []
    codecs = []
    if intra_enabled:
        stages.extend(["Compress(I)", "AllToAll(intra)"])
        if inter_enabled:
            stages.append("DRC(I->E,L)")
            codecs.append(f"I->E:{drc_mode(intra_codec, inter_codec)}")
        else:
            stages.extend(["DR(I,L)", "NativeReduceScatter(inter)"])
    else:
        stages.append("NativeReduceScatter(intra)")
        if inter_enabled:
            stages.extend(["Compress(E)", "AllToAll(inter)"])
    if inter_enabled:
        if operation == "allreduce" and final_enabled:
            stages.append("DRC(E->D,N)")
            codecs.append(f"E->D:{drc_mode(inter_codec, default)}")
        else:
            stages.append("DR(E,N)")
    if operation == "allreduce":
        if final_enabled:
            if not inter_enabled:
                stages.append("Compress(D)")
            stages.extend(["AllGather(default)", "Decompress(D)"])
        else:
            stages.append("RawAllGather")
    return " -> ".join(stages), ";".join(codecs) or "none"


def write_graphs(root, selector_passed):
    rows = []
    for case_name in SCOPES:
        if case_name == "dietgpu-inter":
            continue
        for operation in ("reducescatter", "allreduce"):
            graph, codecs = graph_for(case_name, operation)
            rows.append({"case": case_name, "operation": operation,
                         "graph": graph, "codec_transitions": codecs,
                         "chunk_chain": "8->2->1" +
                         ("->8" if operation == "allreduce" else "")})
    write_csv(root / "pipeline_graphs.csv", rows,
              ["case", "operation", "graph", "codec_transitions",
               "chunk_chain"])
    status = "PASS" if selector_passed else "MISSING"
    selector = [
        {"case": "default-only", "expected": "M18 candidate set preserved",
         "status": status},
        {"case": "intra-only", "expected": "hierarchical candidate eligible",
         "status": status},
        {"case": "inter-only", "expected": "hierarchical candidate eligible",
         "status": status},
        {"case": "missing-scope-model", "expected": "heuristic fallback",
         "status": status},
    ]
    write_csv(root / "selector_cases.csv", selector,
              ["case", "expected", "status"])
    return rows, selector


def load_m17_references(root):
    candidates = [root.parent / "M17" / "correctness.csv",
                  Path("/data/home/scyb672/run/lxc/COCCL-migrate/results/M17/correctness.csv")]
    references = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row["implementation"] != "current":
                    continue
                key = (row["operation"], row["compressor"], row["datatype"],
                       int(row["depth"]))
                references[key] = float(row["mean_relative_error"])
        break
    return references


def parse_correctness(root):
    references = load_m17_references(root)
    rows = []
    for log in sorted((root / "raw" / "correctness").glob("*.log")):
        case_name, depth_text, datatype = log.stem.split("__")
        depth = int(depth_text[1:])
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith("COCCL_CORRECTNESS "):
                continue
            value = fields(line)
            operation = value["operation"]
            error = float(value["mean_relative_error"])
            codecs = ("zfp", "sdp4bit") if case_name == "mixed-sdp-zfp" else (
                ("zfp",) if "zfp" in case_name else ("sdp4bit",))
            candidates = [references.get((operation, codec, datatype, depth))
                          for codec in codecs]
            candidates = [item for item in candidates if item is not None]
            reference = max(candidates) if candidates else 0.0
            floor = {"float": 1e-6, "half": 1e-3,
                     "bfloat16": 1e-2}[datatype]
            limit = max(10.0 * reference, floor)
            rows.append({
                "case": case_name, "operation": operation,
                "datatype": datatype, "depth": depth,
                "mean_relative_error": error,
                "reference_error": reference, "limit": limit,
                "status": "PASS" if math.isfinite(error) and error <= limit
                else "FAIL",
            })
    write_csv(root / "correctness_two_nodes.csv", rows,
              ["case", "operation", "datatype", "depth",
               "mean_relative_error", "reference_error", "limit", "status"])
    return rows


def parse_nccl_perf(log):
    values = []
    for line in log.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) != 13 or not tokens[0].isdigit():
            continue
        values.append((0, float(tokens[5]), float(tokens[6]), float(tokens[7])))
        values.append((1, float(tokens[9]), float(tokens[10]), float(tokens[11])))
    return values


def load_m17_performance(root):
    candidates = [root.parent / "M17" / "performance.csv",
                  Path("/data/home/scyb672/run/lxc/COCCL-migrate/results/M17/performance.csv")]
    result = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row["implementation"] != "current":
                    continue
                key = (row["operation"], row["compressor"], int(row["depth"]),
                       int(row["bytes"]), int(row["inplace"]))
                result[key] = float(row["time_us"])
        break
    return result


def memory_total(root, stem):
    path = root / "memory-samples" / f"{stem}.csv"
    if not path.exists():
        return ""
    per_gpu = {}
    for line in path.read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            key = (parts[0], parts[1])
            per_gpu[key] = max(per_gpu.get(key, 0), int(parts[2]))
    return sum(per_gpu.values()) if len(per_gpu) == 8 else ""


def parse_performance(root):
    m17 = load_m17_performance(root)
    rows = []
    native = {}
    for log in sorted((root / "raw" / "performance").glob("*.log")):
        path, case_name, operation, depth_text, bytes_text = log.stem.split("__")
        depth = int(depth_text[1:])
        byte_count = int(bytes_text[1:])
        for inplace, time_us, algbw, busbw in parse_nccl_perf(log):
            row = {
                "path": path, "case": case_name, "operation": operation,
                "depth": depth, "bytes": byte_count, "inplace": inplace,
                "time_us": time_us, "algbw_gbps": algbw,
                "busbw_gbps": busbw,
                "peak_total_mib": memory_total(root, log.stem),
                "m17_time_us": "", "m17_ratio": "", "native_speedup": "",
                "status": "RECORDED",
            }
            if path == "native":
                native[(operation, byte_count, inplace)] = time_us
            rows.append(row)
    for row in rows:
        if row["path"] == "native":
            continue
        native_time = native.get((row["operation"], row["bytes"], row["inplace"]))
        if native_time is not None:
            row["native_speedup"] = native_time / row["time_us"]
        if row["case"] in ("default-sdp", "default-zfp"):
            compressor = "sdp4bit" if row["case"] == "default-sdp" else "zfp"
            previous = m17.get((row["operation"], compressor, row["depth"],
                                row["bytes"], row["inplace"]))
            if previous is not None:
                row["m17_time_us"] = previous
                row["m17_ratio"] = row["time_us"] / previous
                row["status"] = "PASS" if row["m17_ratio"] <= 1.03 else "FAIL"
    write_csv(root / "performance_two_nodes.csv", rows,
              ["path", "case", "operation", "depth", "bytes", "inplace",
               "time_us", "algbw_gbps", "busbw_gbps", "peak_total_mib",
               "native_speedup", "m17_time_us", "m17_ratio", "status"])
    return rows


def parse_dietgpu(root):
    perf = {}
    correctness = []
    for log in sorted((root / "raw" / "dietgpu-correctness").glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("COCCL_INTEGER_CORRECTNESS "):
                value = fields(line)
                correctness.append({
                    "operation": value["operation"],
                    "depth": int(value["depth"]),
                    "mismatches": int(value["mismatches"]),
                })
    for log in sorted((root / "raw" / "dietgpu-performance").glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith("COCCL_INTEGER_PERF "):
                continue
            value = fields(line)
            key = (value["operation"], value["pattern"], int(value["depth"]),
                   int(value["input_bytes"]))
            perf.setdefault(key, {})[value["path"]] = {
                "latency": float(value["latency_us"]), "stem": log.stem}
    probes = {}
    for log in sorted((root / "raw" / "dietgpu-codec").glob("*.log")):
        operation, pattern, bytes_text = log.stem.split("__")
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("COCCL_DIETGPU_PROBE "):
                value = fields(line)
                probes[(operation, pattern, int(bytes_text[1:]))] = value
    rows = []
    for key, paths in sorted(perf.items()):
        operation, pattern, depth, byte_count = key
        native = paths.get("native")
        compressed = paths.get("compressed")
        probe = probes.get((operation, pattern, byte_count), {})
        encode = float(probe.get("encode_us", 0.0))
        decode = float(probe.get("decode_us", 0.0))
        compressed_us = compressed["latency"] if compressed else math.nan
        rows.append({
            "operation": operation, "pattern": pattern, "depth": depth,
            "raw_bytes": byte_count,
            "payload_bytes": probe.get("payload_bytes", ""),
            "compression_ratio": probe.get("ratio", ""),
            "encode_us_first_call": encode or "",
            "decode_us_first_call": decode or "",
            "native_us": native["latency"] if native else "",
            "compressed_us": compressed_us if compressed else "",
            "estimated_non_codec_us": max(0.0, compressed_us - encode - decode)
                if compressed and probe else "",
            "speedup": native["latency"] / compressed_us
                if native and compressed else "",
            "peak_total_mib": memory_total(root, compressed["stem"])
                if compressed else "",
            "status": "PASS" if native and compressed else "MISSING",
        })
    write_csv(root / "dietgpu_inter_only.csv", rows,
              ["operation", "pattern", "depth", "raw_bytes",
               "payload_bytes", "compression_ratio",
               "encode_us_first_call", "decode_us_first_call",
               "estimated_non_codec_us", "native_us", "compressed_us",
               "speedup", "peak_total_mib", "status"])
    return correctness, rows


def write_workspace(root, performance, dietgpu):
    rows = []
    for row in performance:
        if row["peak_total_mib"] != "":
            rows.append({key: row[key] for key in
                         ("case", "operation", "depth", "bytes",
                          "peak_total_mib")})
    for row in dietgpu:
        if row["peak_total_mib"] != "":
            rows.append({"case": "dietgpu-inter", "operation": row["operation"],
                         "depth": row["depth"], "bytes": row["raw_bytes"],
                         "peak_total_mib": row["peak_total_mib"]})
    write_csv(root / "workspace.csv", rows,
              ["case", "operation", "depth", "bytes", "peak_total_mib"])


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "build-manifest.txt"
    selector_passed = (manifest.exists() and
                       "host_tests=M19 PASS" in manifest.read_text())
    effective = write_scope_tables(root)
    graphs, selector = write_graphs(root, selector_passed)
    correctness = parse_correctness(root)
    performance = parse_performance(root)
    diet_correctness, dietgpu = parse_dietgpu(root)
    write_workspace(root, performance, dietgpu)

    failed_markers = list((root / "raw").glob("**/*.failed"))
    correctness_failures = [row for row in correctness if row["status"] == "FAIL"]
    perf_failures = [row for row in performance if row["status"] == "FAIL"]
    diet_lossless_failures = [row for row in diet_correctness
                              if row["mismatches"] != 0]
    diet_missing = [row for row in dietgpu if row["status"] != "PASS"]
    selector_missing = [row for row in selector if row["status"] != "PASS"]
    incomplete = []
    for label, actual, expected in (
            ("correctness", len(correctness), EXPECTED_CORRECTNESS_ROWS),
            ("performance", len(performance), EXPECTED_PERFORMANCE_ROWS),
            ("dietgpu correctness", len(diet_correctness),
             EXPECTED_DIETGPU_CORRECTNESS_ROWS),
            ("dietgpu performance", len(dietgpu),
             EXPECTED_DIETGPU_PERFORMANCE_ROWS)):
        if actual != expected:
            incomplete.append(f"{label}: {actual}/{expected}")
    passed = not (failed_markers or correctness_failures or perf_failures or
                  diet_lossless_failures or diet_missing or selector_missing or
                  incomplete)

    onset = {}
    for row in dietgpu:
        if row["status"] == "PASS" and row["speedup"] > 1.0:
            key = (row["operation"], row["pattern"], row["depth"])
            onset[key] = min(onset.get(key, row["raw_bytes"]), row["raw_bytes"])

    with (root / "report.md").open("w") as stream:
        stream.write("# M19 Default/Intra/Inter 分层策略报告\n\n")
        stream.write(f"- Result: {'PASS' if passed else 'FAIL'}\n")
        stream.write(f"- Effective scope cases: {len(effective)}\n")
        stream.write(f"- Floating-point correctness rows: {len(correctness)}; "
                     f"failures: {len(correctness_failures)}\n")
        stream.write(f"- Performance rows: {len(performance)}; M17 regression "
                     f"failures: {len(perf_failures)}\n")
        stream.write(f"- dietGPU lossless rows: {len(diet_correctness)}; "
                     f"mismatch failures: {len(diet_lossless_failures)}\n")
        stream.write(f"- dietGPU performance pairs: {len(dietgpu)}; "
                     f"missing pairs: {len(diet_missing)}\n")
        stream.write(f"- Selector Host cases: {len(selector)}; "
                     f"missing/failures: {len(selector_missing)}\n")
        if incomplete:
            stream.write(f"- Incomplete matrix: {', '.join(incomplete)}\n")

        stream.write("\n## Scope 配置与有效策略\n\n")
        policy_rows = []
        for row in effective:
            explicit = SCOPES[row["case"]]
            policy_rows.append((row["case"], *explicit, row["default"],
                                row["intra"], row["inter"]))
        markdown_table(
            stream,
            ["case", "explicit D", "explicit I", "explicit E",
             "effective D", "effective I", "effective E"],
            policy_rows)

        stream.write("\n## Pipeline Graph\n\n")
        stream.write("这些图由与 Host graph test 相同的 scope 解析规则生成；"
                     "算法固定为 ReduceScatter TwoShot 或 AllReduce TripleShot。\n\n")
        markdown_table(
            stream,
            ["case", "algorithm", "stages", "DRC", "chunks"],
            ((row["case"],
              "RS TwoShot" if row["operation"] == "reducescatter"
              else "AR TripleShot",
              row["graph"], row["codec_transitions"], row["chunk_chain"])
             for row in graphs))

        stream.write("\n## 正确性\n\n")
        correctness_groups = {}
        for row in correctness:
            key = (row["case"], row["operation"])
            correctness_groups.setdefault(key, []).append(
                float(row["mean_relative_error"]))
        markdown_table(
            stream,
            ["case", "algorithm", "rows", "mean relative error",
             "max relative error"],
            ((case, "RS TwoShot" if operation == "reducescatter"
              else "AR TripleShot", len(values),
              f"{sum(values) / len(values):.6e}", f"{max(values):.6e}")
             for (case, operation), values in sorted(
                 correctness_groups.items())))

        stream.write("\n## 性能与显存\n\n")
        stream.write("下表为 out-of-place 结果；单点格式为 `time_us / native speedup`。"
                     "`peak MiB` 是该 case 所有消息点中 8 卡总显存采样峰值，"
                     "精确 planner workspace 由 Host plan tests 验证。"
                     "In-place 原始数据保存在 `performance_two_nodes.csv`。\n\n")
        perf_groups = {}
        for row in performance:
            if row["path"] != "compressed" or row["inplace"] != 0:
                continue
            key = (row["case"], row["operation"], row["depth"])
            perf_groups.setdefault(key, {})[row["bytes"]] = row
        perf_rows = []
        for key, points in sorted(perf_groups.items()):
            cells = []
            for byte_count in (67108864, 536870912, 1073741824):
                point = points[byte_count]
                cells.append(f"{point['time_us']:.1f} / "
                             f"{point['native_speedup']:.3f}x")
            peak = max(int(point["peak_total_mib"])
                       for point in points.values())
            ratios = [float(point["m17_ratio"]) for point in points.values()
                      if point["m17_ratio"] != ""]
            perf_rows.append((*key, *cells, peak,
                              f"{max(ratios):.4f}" if ratios else "n/a"))
        markdown_table(
            stream,
            ["case", "operation", "depth", "64 MiB", "512 MiB", "1 GiB",
             "peak MiB", "max M17 ratio"], perf_rows)

        stream.write("\n## Selector\n\n")
        markdown_table(stream, ["case", "expected", "status"],
                       ((row["case"], row["expected"], row["status"])
                        for row in selector))

        stream.write("\n## dietGPU 纯节点间压缩\n\n")
        markdown_table(
            stream,
            ["operation", "pattern", "depth", "raw MiB", "payload bytes",
             "ratio", "native us", "compressed us", "speedup", "peak MiB"],
            ((row["operation"], row["pattern"], row["depth"],
              int(row["raw_bytes"]) // (1024 * 1024), row["payload_bytes"],
              row["compression_ratio"], f"{row['native_us']:.3f}",
              f"{row['compressed_us']:.3f}", f"{row['speedup']:.3f}x",
              row["peak_total_mib"]) for row in dietgpu))

        stream.write("\nLossless correctness:\n\n")
        markdown_table(
            stream, ["operation", "depth", "mismatches"],
            ((row["operation"], row["depth"], row["mismatches"])
             for row in diet_correctness))
        stream.write("\n## dietGPU 加速起点\n\n")
        if not onset:
            stream.write("在本次消息大小矩阵中没有端到端加速点。\n")
        else:
            for key, byte_count in sorted(onset.items()):
                stream.write(f"- {key[0]}, {key[1]}, depth={key[2]}: "
                             f"{byte_count} bytes\n")
        stream.write("\nCodec probe 时间是独立进程的首次调用时间；"
                     "`estimated_non_codec_us` 是端到端时间减去该值后的观测残差，"
                     "包含通信、归约、final AllGather 和调度，不作为独立通信内核计时。\n")
        if failed_markers:
            stream.write("\n## Failed cases\n\n")
            for marker in failed_markers:
                stream.write(f"- `{marker}`\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
