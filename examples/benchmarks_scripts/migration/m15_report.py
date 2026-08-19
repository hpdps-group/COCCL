#!/usr/bin/env python3

import csv
import sys
from pathlib import Path


MIN_GATED_BYTES = 512 * 1024 * 1024
MAX_REGRESSION_PERCENT = 3.0


def key(row):
    return int(row["bytes"]), int(row["depth"]), row["mode"]


def main():
    result_root = Path(sys.argv[1])
    raw_path = result_root / "layout_performance_raw.csv"
    with raw_path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))

    expected_rows = 3 * 4 * 5
    baseline = {
        key(row): float(row["time_us"])
        for row in raw if row["version"] == "m14"
    }
    current_plain = {
        key(row): float(row["time_us"])
        for row in raw
        if row["version"] == "m15" and row["mode"] == "plain-pack"
    }
    output = []
    gated = []
    for row in raw:
        reference = None
        reference_name = ""
        if row["version"] == "m15" and row["mode"] in (
                "plain-pack", "plain-unpack"):
            reference = baseline[key(row)]
            reference_name = "m14-" + row["mode"]
        elif row["version"] == "m15" and row["mode"] == "swizzle":
            plain_key = (int(row["bytes"]), int(row["depth"]), "plain-pack")
            reference = current_plain[plain_key]
            reference_name = "m15-plain-pack"

        regression = ""
        passed = ""
        if reference is not None:
            regression_value = (
                float(row["time_us"]) / reference - 1.0) * 100.0
            regression = f"{regression_value:.6f}"
            if int(row["bytes"]) >= MIN_GATED_BYTES:
                passed = "PASS" if regression_value <= MAX_REGRESSION_PERCENT else "FAIL"
                gated.append(regression_value)
            else:
                passed = "RECORDED"
        output.append({
            **row,
            "reference": reference_name,
            "reference_us": "" if reference is None else f"{reference:.6f}",
            "regression_percent": regression,
            "gate": passed,
        })

    fields = list(output[0])
    with (result_root / "pack_performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    layout_ok = "PASS" in (result_root / "gpu-layout.txt").read_text()
    path_rows = list(csv.DictReader(
        (result_root / "path_selection.csv").open(newline="")))
    path_ok = len(path_rows) == 12 and all(
        row["status"] == "PASS" for row in path_rows)
    performance_ok = (
        len(raw) == expected_rows and len(gated) == 24 and
        max(gated) <= MAX_REGRESSION_PERCENT
    )
    passed = layout_ok and path_ok and performance_ok

    max_regression = max(gated)
    with (result_root / "profiler_summary.md").open("w") as stream:
        stream.write("# M15 Layout Kernel Profile Summary\n\n")
        stream.write("- CUDA event timing: warmup 20, iterations 30.\n")
        stream.write("- Each size/depth/mode is measured in a new process.\n")
        stream.write(f"- Maximum gated regression: {max_regression:.3f}%.\n")
        stream.write("- The optimized Plain kernels retain the M14 vector paths; ")
        stream.write("Swizzle adds one chunk-index transform per row.\n")
        stream.write("- SDP4Bit advertises fused Swizzle and therefore keeps the ")
        stream.write("measured Plain Pack path at every depth.\n")

    with (result_root / "report.md").open("w") as stream:
        stream.write("# M15 Pack/Unpack Swizzle Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write(f"- GPU layout cases passed: {layout_ok}.\n")
        stream.write(f"- Compressor path selections passed: {path_ok}.\n")
        stream.write(f"- Performance rows: {len(raw)}/{expected_rows}.\n")
        stream.write(f"- Maximum regression at 512 MiB or larger: ")
        stream.write(f"{max_regression:.3f}% (limit {MAX_REGRESSION_PERCENT:.1f}%).\n")
        stream.write("- Node-local RS TwoShot/AR TripleShot are outside the benchmark ")
        stream.write("matrix; real SDP4Bit hierarchical execution is checked by the ")
        stream.write("M16 2x4-node matrix.\n")
        stream.write("- M16 owns end-to-end collective numerical validation.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
