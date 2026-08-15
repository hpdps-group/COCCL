#!/usr/bin/env python3

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_PERFORMANCE_PER_VARIANT = 1080
EXPECTED_DATA_ROWS_PER_VARIANT = 1080
EXPECTED_MEMORY_PER_VARIANT = 48
PERFORMANCE_GATE_MIN_BYTES = 1 << 29


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0],
        "algorithm": parts[1],
        "compressor": parts[2],
        "depth": int(parts[3][1:]),
        "bytes": int(parts[4][1:]),
        "run_id": int(parts[5][1:]),
    }


def parse_performance_log(path, variant, case):
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue

            base = {
                "variant": variant,
                **case,
                "bytes": int(tokens[0]),
            }
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
    fieldnames = [
        "variant",
        "collective",
        "algorithm",
        "compressor",
        "depth",
        "bytes",
        "run_id",
        "inplace",
        "time_us",
        "algbw_gbps",
        "busbw_gbps",
    ]
    rows = []
    process_counts = {}
    data_row_counts = {}
    for variant in ("original", "migrate-copy"):
        directory = result_root / "raw" / "current" / variant
        markers = sorted(directory.glob("*__*.ok"))
        process_counts[variant] = len(markers)
        variant_rows = []
        for marker in markers:
            log = marker.with_suffix(".log")
            variant_rows.extend(
                parse_performance_log(log, variant, case_fields(marker))
            )
        data_row_counts[variant] = len(variant_rows) // 2
        rows.extend(variant_rows)

    with (result_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["variant"],
            row["collective"],
            row["algorithm"],
            row["compressor"],
            row["depth"],
            row["bytes"],
            row["inplace"],
        )
        grouped[key].append(row["time_us"])

    medians = {key: statistics.median(values) for key, values in grouped.items()}
    comparison_fields = [
        "collective",
        "algorithm",
        "compressor",
        "depth",
        "bytes",
        "inplace",
        "original_median_us",
        "migrate_median_us",
        "relative_delta",
        "is_gate_point",
        "within_3_percent",
    ]
    comparisons = []
    original_keys = sorted(key for key in medians if key[0] == "original")
    for key in original_keys:
        migrate_key = ("migrate-copy", *key[1:])
        if migrate_key not in medians:
            continue
        original_time = medians[key]
        migrate_time = medians[migrate_key]
        delta = (migrate_time - original_time) / original_time
        comparisons.append(
            {
                "collective": key[1],
                "algorithm": key[2],
                "compressor": key[3],
                "depth": key[4],
                "bytes": key[5],
                "inplace": key[6],
                "original_median_us": original_time,
                "migrate_median_us": migrate_time,
                "relative_delta": delta,
                "is_gate_point": int(key[5] >= PERFORMANCE_GATE_MIN_BYTES),
                "within_3_percent": int(abs(delta) <= 0.03),
            }
        )

    with (result_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)

    return process_counts, data_row_counts, len(rows), len(comparisons)


def parse_memory_sample(path):
    peak_by_gpu = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            gpu_uuid = fields[2]
            used_mib = int(fields[3])
            peak_by_gpu[gpu_uuid] = max(peak_by_gpu[gpu_uuid], used_mib)
    return sum(peak_by_gpu.values()), len(peak_by_gpu)


def write_memory(result_root):
    fields = [
        "variant",
        "collective",
        "algorithm",
        "compressor",
        "depth",
        "bytes",
        "run_id",
        "peak_total_mib",
        "sampled_gpus",
    ]
    rows = []
    counts = {}
    for variant in ("original", "migrate-copy"):
        sample_dir = result_root / "memory-samples" / "current" / variant
        marker_dir = result_root / "raw" / "current-memory" / variant
        counts[variant] = len(list(marker_dir.glob("*.ok")))
        for sample in sorted(sample_dir.glob("*.csv")):
            marker = marker_dir / (sample.stem + ".ok")
            if not marker.exists():
                continue
            peak, gpu_count = parse_memory_sample(sample)
            rows.append(
                {
                    "variant": variant,
                    **case_fields(sample),
                    "peak_total_mib": peak,
                    "sampled_gpus": gpu_count,
                }
            )

    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return counts, len(rows)


def main():
    result_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/data/home/scyb672/run/lxc/COCCL-migrate/results/M0"
    )
    process_counts, data_row_counts, performance_rows, comparison_rows = (
        write_performance(result_root)
    )
    memory_counts, memory_rows = write_memory(result_root)

    with (result_root / "completeness.txt").open("w") as stream:
        stream.write(
            "expected_performance_per_variant="
            f"{EXPECTED_PERFORMANCE_PER_VARIANT}\n"
        )
        for variant, count in process_counts.items():
            stream.write(f"performance_{variant}={count}\n")
        stream.write(
            "expected_data_rows_per_variant="
            f"{EXPECTED_DATA_ROWS_PER_VARIANT}\n"
        )
        for variant, count in data_row_counts.items():
            stream.write(f"data_rows_{variant}={count}\n")
        stream.write(f"parsed_performance_rows={performance_rows}\n")
        stream.write(f"comparison_rows={comparison_rows}\n")
        stream.write(f"performance_gate_min_bytes={PERFORMANCE_GATE_MIN_BYTES}\n")
        stream.write(f"expected_memory_per_variant={EXPECTED_MEMORY_PER_VARIANT}\n")
        for variant, count in memory_counts.items():
            stream.write(f"memory_{variant}={count}\n")
        stream.write(f"parsed_memory_rows={memory_rows}\n")


if __name__ == "__main__":
    main()
