#!/usr/bin/env python3

import csv
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


MODEL_NUMBER = r"([-+0-9.eE]+)"
P2P_RE = re.compile(
    rf"COCCL profile (intra|inter) P2P: time_us={MODEL_NUMBER}\+{MODEL_NUMBER}\*bytes"
)
CODEC_RE = re.compile(
    rf"COCCL compressor (\w+) policy=(reducescatter|allreduce)-(default|hierarchical) model: time_us={MODEL_NUMBER}\+{MODEL_NUMBER}\*bytes ratio={MODEL_NUMBER}"
)
SELECT_RE = re.compile(
    rf"COCCL select bytes={MODEL_NUMBER} ranks=(\d+) local=(\d+) nodes=(\d+) .* -> (\w+)"
)
ORACLE_DEPTH = 1


def parse_profile(result_root, compressor):
    log = result_root / "raw" / "profile" / f"{compressor}.log"
    text = log.read_text(errors="replace")
    p2p = {}
    for match in P2P_RE.finditer(text):
        p2p[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    policies = {}
    for match in CODEC_RE.finditer(text):
        if match.group(1) == compressor:
            policies[f"{match.group(2)}-{match.group(3)}"] = {
                "codec_alpha_us": float(match.group(4)),
                "codec_beta_us_per_byte": float(match.group(5)),
                "compression_ratio": float(match.group(6)),
            }
    selections = [
        {
            "bytes": int(float(match.group(1))),
            "ranks": int(match.group(2)),
            "local_ranks": int(match.group(3)),
            "nodes": int(match.group(4)),
            "algorithm": match.group(5),
        }
        for match in SELECT_RE.finditer(text)
    ]
    required = {
        "reducescatter-default", "reducescatter-hierarchical",
        "allreduce-default", "allreduce-hierarchical",
    }
    if set(p2p) != {"intra", "inter"} or not required.issubset(policies) or not selections:
        raise RuntimeError(f"incomplete autotune profile in {log}")
    return {
        "compressor": compressor,
        "intra_alpha_us": p2p["intra"][0],
        "intra_beta_us_per_byte": p2p["intra"][1],
        "inter_alpha_us": p2p["inter"][0],
        "inter_beta_us_per_byte": p2p["inter"][1],
        "policies": policies,
        "profile_min_bytes": 3145728,
        "profile_max_bytes": 201326592,
        "pipeline_depth": 1,
        "runtime_selected_algorithm": selections[-1]["algorithm"],
        "log": str(log),
    }


def write_snapshots(result_root):
    snapshots = {
        compressor: parse_profile(result_root, compressor)
        for compressor in ("sdp4bit", "zfp")
    }
    fields = [
        "compressor", "operation", "variant", "intra_alpha_us",
        "intra_beta_us_per_byte", "inter_alpha_us",
        "inter_beta_us_per_byte", "codec_alpha_us",
        "codec_beta_us_per_byte", "compression_ratio",
        "profile_min_bytes", "profile_max_bytes", "pipeline_depth",
        "runtime_selected_algorithm", "log",
    ]
    rows = []
    for compressor, snapshot in snapshots.items():
        for policy, codec in sorted(snapshot["policies"].items()):
            operation, variant = policy.split("-", 1)
            rows.append(
                {
                    "compressor": compressor,
                    "operation": operation,
                    "variant": variant,
                    **{key: snapshot[key] for key in (
                        "intra_alpha_us", "intra_beta_us_per_byte",
                        "inter_alpha_us", "inter_beta_us_per_byte",
                        "profile_min_bytes", "profile_max_bytes",
                        "pipeline_depth", "runtime_selected_algorithm", "log",
                    )},
                    **codec,
                }
            )
    combined = result_root / "model_snapshots" / "models.csv"
    with combined.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for compressor in snapshots:
        with (result_root / "model_snapshots" / f"{compressor}.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row["compressor"] == compressor)
    return snapshots


def read_depth_one_oracle(path):
    all_rows = list(csv.DictReader(path.open()))
    rows = [row for row in all_rows if int(row["depth"]) == ORACLE_DEPTH]
    if not rows or any(int(row["depth"]) != ORACLE_DEPTH for row in rows):
        raise RuntimeError("M18 oracle must contain depth=1 rows only")
    return all_rows, rows


def write_candidate_timings(result_root, rows):
    fields = [
        "topology", "rank_count", "operation", "compressor", "datatype",
        "depth", "bytes", "candidate", "elapsed_time_us",
        "measured_best_time_us", "measured_best_set", "within_best_set", "log",
    ]
    with (result_root / "candidate_timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_model_selector(binary, grouped, snapshots):
    lines = [
        "case_id,operation,compressor,bytes,local_ranks,nodes,intra_alpha,intra_beta,inter_alpha,inter_beta,base_codec_alpha,base_codec_beta,base_compression_ratio,hierarchical_codec_alpha,hierarchical_codec_beta,hierarchical_compression_ratio"
    ]
    for case_id, key in enumerate(sorted(grouped)):
        operation, compressor, bytes_value = key
        snapshot = snapshots[compressor]
        base = snapshot["policies"][f"{operation}-default"]
        hierarchical = snapshot["policies"].get(
            f"{operation}-hierarchical", base
        )
        lines.append(
            ",".join(
                map(
                    str,
                    [
                        case_id, operation, compressor, bytes_value, 4, 2,
                        snapshot["intra_alpha_us"],
                        snapshot["intra_beta_us_per_byte"],
                        snapshot["inter_alpha_us"],
                        snapshot["inter_beta_us_per_byte"],
                        base["codec_alpha_us"],
                        base["codec_beta_us_per_byte"],
                        base["compression_ratio"],
                        hierarchical["codec_alpha_us"],
                        hierarchical["codec_beta_us_per_byte"],
                        hierarchical["compression_ratio"],
                    ],
                )
            )
        )
    process = subprocess.run(
        [str(binary)], input="\n".join(lines) + "\n", text=True,
        capture_output=True, check=True
    )
    return {
        (row["operation"], row["compressor"], int(row["bytes"])): row
        for row in csv.DictReader(process.stdout.splitlines())
    }


def write_optimality(result_root, oracle_rows, snapshots, selector_binary):
    grouped = defaultdict(list)
    for row in oracle_rows:
        grouped[(row["operation"], row["compressor"], int(row["bytes"]))].append(row)
    modeled = run_model_selector(selector_binary, grouped, snapshots)

    output = []
    for key in sorted(grouped):
        rows = grouped[key]
        model = modeled[key]
        timings = {
            row["candidate"]: float(row["elapsed_time_us"]) for row in rows
        }
        selected = model["selected_algorithm"]
        best_time = min(timings.values())
        best_set = rows[0]["measured_best_set"].split(";")
        selected_time = timings.get(selected)
        correct = selected in best_set
        status = "PASS" if correct and selected_time is not None else "FAIL"
        output.append(
            {
                "operation": key[0],
                "compressor": key[1],
                "datatype": rows[0]["datatype"],
                "topology": rows[0]["topology"],
                "rank_count": rows[0]["rank_count"],
                "depth": ORACLE_DEPTH,
                "bytes": key[2],
                "eligible_candidates": model["eligible_candidates"],
                "measured_candidates": ";".join(sorted(timings)),
                "candidate_elapsed_us": ";".join(
                    f"{name}={timings[name]}" for name in sorted(timings)
                ),
                "selected_algorithm": selected,
                "measured_best_set": ";".join(best_set),
                "selected_time_us": "" if selected_time is None else selected_time,
                "measured_best_time_us": best_time,
                "slowdown": "" if selected_time is None else (selected_time - best_time) / best_time,
                "selection_correct": int(correct),
                "one_shot_score_us": model["oneshot_us"],
                "two_shot_score_us": model["twoshot_us"],
                "triple_shot_score_us": model["tripleshot_us"],
                "used_model": model["used_model"],
                "status": status,
            }
        )

    fields = list(output[0].keys())
    with (result_root / "optimality_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return output


def write_selector_cases(result_root, optimality):
    cpu_rows = list(csv.DictReader((result_root / "selector_cases_cpu.csv").open()))
    fields = [
        "category", "scenario", "compressor", "operation", "bytes",
        "eligible_candidates", "selected_algorithm", "used_model",
        "forced_fallback", "status",
    ]
    rows = [
        {
            "category": "cpu",
            "scenario": row["scenario"],
            "compressor": "",
            "operation": "",
            "bytes": "",
            "eligible_candidates": row["eligible_candidates"],
            "selected_algorithm": row["selected_algorithm"],
            "used_model": row["used_model"],
            "forced_fallback": row["forced_fallback"],
            "status": row["status"],
        }
        for row in cpu_rows
    ]
    rows.extend(
        {
            "category": "depth1_oracle",
            "scenario": "model_selection",
            "compressor": row["compressor"],
            "operation": row["operation"],
            "bytes": row["bytes"],
            "eligible_candidates": row["eligible_candidates"],
            "selected_algorithm": row["selected_algorithm"],
            "used_model": row["used_model"],
            "forced_fallback": 0,
            "status": row["status"],
        }
        for row in optimality
    )
    with (result_root / "selector_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(result_root, all_oracle, depth_one, optimality, snapshots):
    failed = [row for row in optimality if row["status"] != "PASS"]
    cpu_rows = list(csv.DictReader((result_root / "selector_cases_cpu.csv").open()))
    status = "PASS" if not failed and all(row["status"] == "PASS" for row in cpu_rows) else "FAIL"
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M18 Autotune Decision Report\n\n")
        stream.write(f"Status: {status}\n\n")
        stream.write("- Selector optimality reference is restricted to M17 `depth=1`.\n")
        stream.write("- Pipeline depths 2/4/8 are not selector inputs and are excluded from pass/fail.\n")
        stream.write(f"- M17 oracle rows consumed: {len(depth_one)}/{len(all_oracle)}.\n")
        stream.write(f"- M17 non-depth1 rows ignored: {len(all_oracle) - len(depth_one)}.\n")
        stream.write(f"- CPU decision cases: {len(cpu_rows)}.\n")
        stream.write(f"- Depth-1 optimality cases: {len(optimality)}.\n")
        stream.write(f"- Wrong selections at any tested size: {len(failed)}.\n")
        stream.write("- Profiler samples are 3, 12, 48, and 192 MiB; they are separate from M17 oracle points.\n")
        stream.write("- Predicted score magnitude is diagnostic only; pass/fail checks the selected algorithm.\n")
        for compressor, snapshot in snapshots.items():
            ratios = ", ".join(
                f"{policy}={model['compression_ratio']:.6g}"
                for policy, model in sorted(snapshot["policies"].items())
            )
            stream.write(
                f"- {compressor} ratios: {ratios}; runtime smoke selected="
                f"{snapshot['runtime_selected_algorithm']}.\n"
            )
    if status != "PASS":
        raise SystemExit(1)


def main():
    result_root = Path(sys.argv[1])
    oracle_path = Path(sys.argv[2])
    selector_binary = Path(sys.argv[3])
    snapshots = write_snapshots(result_root)
    all_oracle, depth_one = read_depth_one_oracle(oracle_path)
    write_candidate_timings(result_root, depth_one)
    optimality = write_optimality(
        result_root, depth_one, snapshots, selector_binary
    )
    write_selector_cases(result_root, optimality)
    write_report(result_root, all_oracle, depth_one, optimality, snapshots)


if __name__ == "__main__":
    main()
