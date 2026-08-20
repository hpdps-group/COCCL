#!/usr/bin/env python3

import csv
import re
import statistics
import sys
from pathlib import Path


BASELINES = {
    "qwen25": {"original": 2.434307, "native": 2.436254},
    "qwen3": {"original": 2.333028, "native": 2.334529},
}
DEPTHS = (1, 2, 4, 8)
OVERLAPS = ("off", "on")
FINAL_LOSS_RE = re.compile(
    r"validation loss at iteration 1000 on validation set.*?"
    r"lm loss value:\s*([0-9.eE+-]+)"
)
PERIODIC_LOSS_RE = re.compile(
    r"validation loss at iteration 1000 \|.*?"
    r"lm loss value:\s*([0-9.eE+-]+)"
)
ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*1000.*?elapsed time per iteration \(ms\):\s*"
    r"([0-9.]+).*?number of skipped iterations:\s*(\d+).*?"
    r"number of nan iterations:\s*(\d+)"
)
HEADER_RE = re.compile(r"COCCL_M23_RUN\s+(.*)")
FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")
ROUTE_RE = re.compile(
    r"COCCL route .*?role=(\w+) operation=(\w+).*?compressor=(\w+)"
)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_case(case_root):
    texts = []
    for name in ("node1.log", "node0.log"):
        path = case_root / name
        texts.append(path.read_text(errors="replace") if path.exists() else "")
    text = "\n".join(texts)

    final_losses = [float(value) for value in FINAL_LOSS_RE.findall(text)]
    periodic_losses = [float(value) for value in PERIODIC_LOSS_RE.findall(text)]
    iterations = [
        (int(index), float(time_ms), int(skipped), int(nan))
        for index, time_ms, skipped, nan in ITERATION_RE.findall(text)
    ]
    steady_times = [value[1] for value in iterations if value[0] > 10]
    header_matches = HEADER_RE.findall(text)
    header = dict(FIELD_RE.findall(header_matches[-1])) if header_matches else {}

    route_counts = {}
    for path in (case_root / "nccl").glob("*"):
        for role, operation, compressor in ROUTE_RE.findall(
                path.read_text(errors="replace")):
            key = (role, operation, compressor)
            route_counts[key] = route_counts.get(key, 0) + 1
    dp_all_gather_routes = route_counts.get(
        ("DP", "AllGather", "sdp4bit"), 0)
    dp_reduce_scatter_routes = route_counts.get(
        ("DP", "ReduceScatter", "sdp4bit"), 0)
    unexpected_routes = sum(
        count for (role, _, _), count in route_counts.items() if role != "DP"
    )
    return {
        "final_loss": final_losses[-1] if final_losses else None,
        "periodic_loss": periodic_losses[-1] if periodic_losses else None,
        "iterations": max((value[0] for value in iterations), default=0),
        "nan_iterations": max((value[3] for value in iterations), default=0),
        "skipped_iterations": max((value[2] for value in iterations), default=0),
        "mean_iteration_ms": statistics.fmean(steady_times) if steady_times else None,
        "overlap_grad_flag": "--overlap-grad-reduce" in text,
        "overlap_param_flag": "--overlap-param-gather" in text,
        "header": header,
        "dp_all_gather_routes": dp_all_gather_routes,
        "dp_reduce_scatter_routes": dp_reduce_scatter_routes,
        "unexpected_routes": unexpected_routes,
    }


def value_or_blank(value):
    return "" if value is None else value


def main():
    result_root = Path(sys.argv[1])
    log_root = Path(sys.argv[2])
    commit = sys.argv[3]
    result_root.mkdir(parents=True, exist_ok=True)

    baseline_rows = []
    for model, values in BASELINES.items():
        low = max(values["original"] * 0.99, values["native"] * 0.99)
        high = min(values["original"] * 1.01, values["native"] * 1.01)
        baseline_rows.append({"model": model, **values,
                              "intersection_low": low,
                              "intersection_high": high})
    write_csv(result_root / "baselines.csv", baseline_rows,
              ["model", "original", "native", "intersection_low",
               "intersection_high"])

    rows = []
    runtime_rows = []
    log_paths = []
    for model in ("qwen25", "qwen3"):
        original = BASELINES[model]["original"]
        native = BASELINES[model]["native"]
        for overlap in OVERLAPS:
            for depth in DEPTHS:
                case_root = log_root / model / f"d{depth}-{overlap}"
                parsed = parse_case(case_root)
                loss = parsed["final_loss"]
                original_error = (
                    abs(loss - original) / abs(original) if loss is not None
                    else None
                )
                native_error = (
                    abs(loss - native) / abs(native) if loss is not None
                    else None
                )
                expected_overlap = overlap == "on"
                flags_match = (
                    parsed["overlap_grad_flag"] == expected_overlap and
                    parsed["overlap_param_flag"] == expected_overlap
                )
                header = parsed["header"]
                identity_valid = (
                    header.get("depth") == str(depth) and
                    header.get("overlap") == overlap and
                    bool(header.get("commit")) and
                    bool(header.get("library"))
                )
                completed = (case_root / "complete.ok").exists()
                status = "PASS" if (
                    completed and parsed["iterations"] == 1000 and
                    loss is not None and original_error <= 0.01 and
                    native_error <= 0.01 and
                    parsed["nan_iterations"] == 0 and
                    parsed["skipped_iterations"] == 0 and flags_match and
                    identity_valid and
                    parsed["dp_all_gather_routes"] > 0 and
                    parsed["dp_reduce_scatter_routes"] > 0 and
                    parsed["unexpected_routes"] == 0
                ) else "FAIL"
                row = {
                    "model": model,
                    "depth": depth,
                    "dp_overlap": overlap,
                    "final_validation_loss": value_or_blank(loss),
                    "periodic_validation_loss": value_or_blank(
                        parsed["periodic_loss"]),
                    "relative_to_original": value_or_blank(original_error),
                    "relative_to_native": value_or_blank(native_error),
                    "iterations": parsed["iterations"],
                    "nan_iterations": parsed["nan_iterations"],
                    "skipped_iterations": parsed["skipped_iterations"],
                    "overlap_flags_match": flags_match,
                    "run_identity_valid": identity_valid,
                    "run_commit": header.get("commit", ""),
                    "run_library": header.get("library", ""),
                    "dp_all_gather_routes": parsed["dp_all_gather_routes"],
                    "dp_reduce_scatter_routes":
                        parsed["dp_reduce_scatter_routes"],
                    "non_dp_compressed_routes": parsed["unexpected_routes"],
                    "status": status,
                    "log": str(case_root / "node1.log"),
                }
                rows.append(row)
                runtime_rows.append({
                    "model": model,
                    "depth": depth,
                    "dp_overlap": overlap,
                    "mean_iteration_ms_after_warmup": value_or_blank(
                        parsed["mean_iteration_ms"]),
                    "status": status,
                })
                log_paths.extend(str(case_root / name)
                                 for name in ("node0.log", "node1.log"))

    fields = [
        "model", "depth", "dp_overlap", "final_validation_loss",
        "periodic_validation_loss", "relative_to_original",
        "relative_to_native", "iterations", "nan_iterations",
        "skipped_iterations", "overlap_flags_match",
        "run_identity_valid", "run_commit", "run_library",
        "dp_all_gather_routes",
        "dp_reduce_scatter_routes",
        "non_dp_compressed_routes", "status", "log",
    ]
    write_csv(result_root / "run_matrix.csv", rows, fields)
    write_csv(result_root / "loss_comparison.csv", rows, fields)
    write_csv(result_root / "runtime_summary.csv", runtime_rows,
              ["model", "depth", "dp_overlap",
               "mean_iteration_ms_after_warmup", "status"])
    (result_root / "log_index.txt").write_text("\n".join(log_paths) + "\n")

    passed = sum(row["status"] == "PASS" for row in rows)
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M23 Qwen training convergence\n\n")
        stream.write(f"COCCL commit: `{commit}`\n\n")
        run_commits = sorted({row["run_commit"] for row in rows})
        run_libraries = sorted({row["run_library"] for row in rows})
        stream.write("Recorded run commits: " + ", ".join(
            f"`{value}`" for value in run_commits) + "\n\n")
        stream.write("Loaded library: " + ", ".join(
            f"`{value}`" for value in run_libraries) + "\n\n")
        stream.write(f"Result: **{passed}/{len(rows)} PASS**\n\n")
        stream.write("| Model | Depth | DP overlap | Final val loss | "
                     "vs original | vs native | Routes AG/RS/other | Status |\n")
        stream.write("|---|---:|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            loss = row["final_validation_loss"]
            original_error = row["relative_to_original"]
            native_error = row["relative_to_native"]
            loss_text = f"{loss:.6f}" if loss != "" else "missing"
            original_text = (f"{100 * original_error:.3f}%"
                             if original_error != "" else "missing")
            native_text = (f"{100 * native_error:.3f}%"
                           if native_error != "" else "missing")
            stream.write(
                f"| {row['model']} | {row['depth']} | {row['dp_overlap']} | "
                f"{loss_text} | {original_text} | {native_text} | "
                f"{row['dp_all_gather_routes']}/"
                f"{row['dp_reduce_scatter_routes']}/"
                f"{row['non_dp_compressed_routes']} | {row['status']} |\n"
            )
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
