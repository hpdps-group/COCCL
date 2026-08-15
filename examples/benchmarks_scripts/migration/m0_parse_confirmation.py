#!/usr/bin/env python3

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from m0_parse import case_fields, parse_memory_sample, parse_performance_log


def read_rows(result_root):
    rows = []
    counts = {}
    for variant in ("original", "migrate-copy"):
        directory = (
            result_root / "confirmation" / "current" / "performance" / variant
        )
        markers = sorted(directory.glob("*.ok"))
        counts[variant] = len(markers)
        for marker in markers:
            rows.extend(
                parse_performance_log(
                    marker.with_suffix(".log"), variant, case_fields(marker)
                )
            )
    return rows, counts


def write_memory_confirmation(result_root):
    rows = []
    counts = {}
    for variant in ("original", "migrate-copy"):
        marker_dir = result_root / "confirmation" / "current" / "memory" / variant
        sample_dir = (
            result_root / "confirmation" / "current" / "memory-samples" / variant
        )
        markers = sorted(marker_dir.glob("*.ok"))
        counts[variant] = len(markers)
        for marker in markers:
            sample = sample_dir / (marker.stem + ".csv")
            peak, gpu_count = parse_memory_sample(sample)
            rows.append(
                {
                    "variant": variant,
                    **case_fields(marker),
                    "peak_total_mib": peak,
                    "sampled_gpus": gpu_count,
                }
            )

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
    with (result_root / "memory-confirmation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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
        )
        grouped[key].append(row["peak_total_mib"])

    comparisons = []
    comparison_fields = [
        "collective",
        "algorithm",
        "compressor",
        "depth",
        "bytes",
        "original_median_mib",
        "migrate_median_mib",
        "absolute_delta_mib",
        "measured_noise_mib",
        "within_noise",
    ]
    for key in sorted(key for key in grouped if key[0] == "original"):
        migrate_key = ("migrate-copy", *key[1:])
        if migrate_key not in grouped:
            continue
        original = grouped[key]
        migrate = grouped[migrate_key]
        original_median = statistics.median(original)
        migrate_median = statistics.median(migrate)
        delta = abs(migrate_median - original_median)
        noise = max(
            max(original) - min(original),
            max(migrate) - min(migrate),
            4,
        )
        comparisons.append(
            {
                "collective": key[1],
                "algorithm": key[2],
                "compressor": key[3],
                "depth": key[4],
                "bytes": key[5],
                "original_median_mib": original_median,
                "migrate_median_mib": migrate_median,
                "absolute_delta_mib": delta,
                "measured_noise_mib": noise,
                "within_noise": int(delta <= noise),
            }
        )

    with (result_root / "memory-confirmation-comparison.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return counts, len(comparisons)


def main():
    result_root = Path(sys.argv[1])
    rows, counts = read_rows(result_root)
    fields = [
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
    with (result_root / "confirmation-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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
        "within_3_percent",
    ]
    comparisons = []
    for key in sorted(key for key in medians if key[0] == "original"):
        migrate_key = ("migrate-copy", *key[1:])
        if migrate_key not in medians:
            continue
        original = medians[key]
        migrate = medians[migrate_key]
        delta = (migrate - original) / original
        comparisons.append(
            {
                "collective": key[1],
                "algorithm": key[2],
                "compressor": key[3],
                "depth": key[4],
                "bytes": key[5],
                "inplace": key[6],
                "original_median_us": original,
                "migrate_median_us": migrate,
                "relative_delta": delta,
                "within_3_percent": int(abs(delta) <= 0.03),
            }
        )

    with (result_root / "confirmation-comparison.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)

    memory_counts, memory_comparisons = write_memory_confirmation(result_root)

    print(f"confirmation_original={counts['original']}")
    print(f"confirmation_migrate-copy={counts['migrate-copy']}")
    print(f"confirmation_comparisons={len(comparisons)}")
    print(f"memory_confirmation_original={memory_counts['original']}")
    print(f"memory_confirmation_migrate-copy={memory_counts['migrate-copy']}")
    print(f"memory_confirmation_comparisons={memory_comparisons}")


if __name__ == "__main__":
    main()
