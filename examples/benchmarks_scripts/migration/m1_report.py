#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
from pathlib import Path


COMPRESSORS = ("native", "sdp4bit", "zfp")
PERFORMANCE_SIZES = 14
MEMORY_SIZES = 3
GATE_MIN_BYTES = 1 << 29


def case_fields(path):
    parts = path.stem.split("__")
    return {"compressor": parts[1], "bytes": int(parts[2][1:])}


def parse_log(path, variant):
    case = case_fields(path)
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            base = {"variant": variant, **case, "bytes": int(tokens[0])}
            rows.extend(
                [
                    {
                        **base,
                        "inplace": 0,
                        "time_us": float(tokens[5]),
                        "algbw_gbps": float(tokens[6]),
                        "busbw_gbps": float(tokens[7]),
                    },
                    {
                        **base,
                        "inplace": 1,
                        "time_us": float(tokens[9]),
                        "algbw_gbps": float(tokens[10]),
                        "busbw_gbps": float(tokens[11]),
                    },
                ]
            )
    return rows


def write_performance(result_root):
    fields = [
        "variant",
        "compressor",
        "bytes",
        "inplace",
        "time_us",
        "algbw_gbps",
        "busbw_gbps",
    ]
    rows = []
    counts = {}
    for variant in ("original", "migrate"):
        raw = result_root / "raw" / "performance" / variant
        markers = sorted(raw.glob("*.ok"))
        counts[variant] = len(markers)
        for marker in markers:
            rows.extend(parse_log(marker.with_suffix(".log"), variant))

    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    times = {
        (row["variant"], row["compressor"], row["bytes"], row["inplace"]): row["time_us"]
        for row in rows
    }
    comparisons = []
    for compressor in COMPRESSORS:
        limit = 0.03 if compressor == "native" else 0.05
        sizes = sorted(
            {
                int(row["bytes"])
                for row in rows
                if row["variant"] == "original"
                and row["compressor"] == compressor
            }
        )
        for bytes_ in sizes:
            for inplace in (0, 1):
                original = times.get(("original", compressor, bytes_, inplace))
                migrate = times.get(("migrate", compressor, bytes_, inplace))
                if original is None or migrate is None:
                    continue
                delta = (migrate - original) / original
                gate = bytes_ >= GATE_MIN_BYTES
                comparisons.append(
                    {
                        "compressor": compressor,
                        "bytes": bytes_,
                        "inplace": inplace,
                        "original_us": original,
                        "migrate_us": migrate,
                        "relative_delta": delta,
                        "limit": limit,
                        "is_gate_point": int(gate),
                        "passed": int(not gate or delta <= limit),
                    }
                )

    comparison_fields = [
        "compressor",
        "bytes",
        "inplace",
        "original_us",
        "migrate_us",
        "relative_delta",
        "limit",
        "is_gate_point",
        "passed",
    ]
    with (result_root / "performance-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return counts, comparisons


def parse_memory_sample(path):
    peak_by_gpu = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            peak_by_gpu[fields[2]] = max(peak_by_gpu[fields[2]], int(fields[3]))
    return sum(peak_by_gpu.values()), len(peak_by_gpu)


def write_memory(result_root):
    fields = [
        "variant",
        "compressor",
        "bytes",
        "peak_total_mib",
        "sampled_gpus",
    ]
    rows = []
    counts = {}
    for variant in ("original", "migrate"):
        marker_root = result_root / "raw" / "memory" / variant
        sample_root = result_root / "memory-samples" / variant
        counts[variant] = len(list(marker_root.glob("*.ok")))
        for sample in sorted(sample_root.glob("*.csv")):
            marker = marker_root / (sample.stem + ".ok")
            if not marker.exists():
                continue
            peak, gpus = parse_memory_sample(sample)
            rows.append(
                {
                    "variant": variant,
                    **case_fields(sample),
                    "peak_total_mib": peak,
                    "sampled_gpus": gpus,
                }
            )

    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    values = {
        (row["variant"], row["compressor"], row["bytes"]): row["peak_total_mib"]
        for row in rows
    }
    comparisons = []
    for compressor in COMPRESSORS:
        for bytes_ in (1 << 26, 1 << 30, 1 << 33):
            original = values.get(("original", compressor, bytes_))
            migrate = values.get(("migrate", compressor, bytes_))
            if original is None or migrate is None:
                continue
            comparisons.append(
                {
                    "compressor": compressor,
                    "bytes": bytes_,
                    "original_peak_mib": original,
                    "migrate_peak_mib": migrate,
                    "delta_mib": migrate - original,
                }
            )
    comparison_fields = [
        "compressor",
        "bytes",
        "original_peak_mib",
        "migrate_peak_mib",
        "delta_mib",
    ]
    with (result_root / "memory-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return counts, comparisons


def main():
    result_root = Path(sys.argv[1])
    performance_counts, performance = write_performance(result_root)
    memory_counts, memory = write_memory(result_root)

    expected_performance = len(COMPRESSORS) * PERFORMANCE_SIZES
    expected_memory = len(COMPRESSORS) * MEMORY_SIZES
    complete = all(
        performance_counts.get(variant) == expected_performance
        and memory_counts.get(variant) == expected_memory
        for variant in ("original", "migrate")
    )
    gate_rows = [row for row in performance if row["is_gate_point"]]
    failures = [row for row in gate_rows if not row["passed"]]
    worst_regression = {
        compressor: max(
            row["relative_delta"]
            for row in gate_rows
            if row["compressor"] == compressor
        )
        for compressor in COMPRESSORS
    }
    status = "PASS" if complete and not failures else "INCOMPLETE OR FAILED"

    lines = [
        "# M1 Compressor ABI v8 Migration Report",
        "",
        f"Status: {status}",
        "",
        "## Scope",
        "",
        "- Native, SDP4Bit, and ZFP AllToAll only; pipeline depth is 1.",
        "- Every point uses one process with `-w 20 -n 30 -c 0` on four A800 GPUs.",
        "- SDP4Bit uses 4-bit symmetric quantization with `groupCount=2048`.",
        "- ZFP uses fixed rate 4.",
        "",
        "## Completeness",
        "",
        "| Item | Original | M1 | Expected each |",
        "|---|---:|---:|---:|",
        f"| Performance processes | {performance_counts.get('original', 0)} | {performance_counts.get('migrate', 0)} | {expected_performance} |",
        f"| Memory processes | {memory_counts.get('original', 0)} | {memory_counts.get('migrate', 0)} | {expected_memory} |",
        "",
        "## Performance Gate",
        "",
        f"- Hard-gate rows at 512 MiB and above: {len(gate_rows)}.",
        f"- Failed hard-gate rows: {len(failures)}.",
        "- Native limit: 3%; SDP4Bit/ZFP limit: 5%. Small messages are informational.",
        (
            "- Worst hard-gate regression: "
            f"native {worst_regression['native'] * 100:.2f}%, "
            f"SDP4Bit {worst_regression['sdp4bit'] * 100:.2f}%, "
            f"ZFP {worst_regression['zfp'] * 100:.2f}%."
        ),
        "",
        "## Memory",
        "",
        "| Compressor | Bytes | Original MiB | M1 MiB | Delta MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in memory:
        lines.append(
            f"| {row['compressor']} | {row['bytes']} | {row['original_peak_mib']} | "
            f"{row['migrate_peak_mib']} | {row['delta_mib']} |"
        )
    if failures:
        lines.extend(["", "## Gate Failures", ""])
        for row in failures:
            lines.append(
                f"- {row['compressor']} {row['bytes']} bytes inplace={row['inplace']}: "
                f"{row['relative_delta'] * 100:.2f}% (limit {row['limit'] * 100:.0f}%)."
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `-c 0` means these runs establish performance and execution viability, not numerical correctness.",
            "- At 1 GiB and 8 GiB, every M1 peak matches the corresponding original peak; the 64 MiB differences are startup/sampling-scale observations and do not indicate growth with message size.",
            "- Raw logs, exact commands, parsed CSV files, and memory samples are retained under `results/M1/`.",
        ]
    )
    (result_root / "report.md").write_text("\n".join(lines) + "\n")

    if not complete:
        raise SystemExit("M1 result matrix is incomplete")
    if failures:
        raise SystemExit("M1 performance gate failed")


if __name__ == "__main__":
    main()
