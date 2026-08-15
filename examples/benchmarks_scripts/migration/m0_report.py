#!/usr/bin/env python3

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read_key_values(path):
    values = {}
    with path.open() as stream:
        for line in stream:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                values[key] = value
    return values


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def row_count(path):
    return len(read_csv(path)) if path.exists() else 0


def performance_key(row):
    return (
        row["collective"],
        row["algorithm"],
        row["compressor"],
        row["depth"],
        row["bytes"],
        row["inplace"],
    )


def performance_summary(rows, confirmation_rows):
    if not rows:
        return None
    deltas = [float(row["relative_delta"]) for row in rows]
    all_failures = [row for row in rows if row["within_3_percent"] != "1"]
    gate_rows = [row for row in rows if row["is_gate_point"] == "1"]
    initial_gate_failures = [
        row for row in gate_rows if row["within_3_percent"] != "1"
    ]
    confirmations = {
        performance_key(row): row for row in confirmation_rows
    }
    effective_gate_rows = []
    confirmed_candidates = 0
    for row in gate_rows:
        if row["within_3_percent"] != "1":
            confirmation = confirmations.get(performance_key(row))
            if confirmation is not None:
                row = confirmation
                confirmed_candidates += 1
        effective_gate_rows.append(row)

    gate_failures = [
        row
        for row in effective_gate_rows
        if row["within_3_percent"] != "1"
    ]
    informational_failures = [
        row
        for row in rows
        if row["is_gate_point"] != "1"
        and row["within_3_percent"] != "1"
    ]
    worst = max(
        effective_gate_rows, key=lambda row: float(row["relative_delta"])
    )
    return {
        "points": len(rows),
        "all_failures": len(all_failures),
        "gate_points": len(gate_rows),
        "initial_gate_failures": len(initial_gate_failures),
        "confirmed_candidates": confirmed_candidates,
        "gate_failures": len(gate_failures),
        "informational_failures": len(informational_failures),
        "max_regression": float(worst["relative_delta"]),
        "max_regression_case": (
            f"{worst['collective']}/{worst['algorithm']}/"
            f"{worst['compressor']}/d{worst['depth']}/"
            f"{worst['bytes']}B/inplace={worst['inplace']}"
        ),
        "median_abs_delta": statistics.median(abs(value) for value in deltas),
    }


def write_memory_comparison(result_root, rows):
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
        grouped[key].append(int(row["peak_total_mib"]))

    fields = [
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
    comparisons = []
    for key, original_values in sorted(grouped.items()):
        if key[0] != "original":
            continue
        migrate_key = ("migrate-copy", *key[1:])
        if migrate_key not in grouped:
            continue
        migrate_values = grouped[migrate_key]
        original_median = statistics.median(original_values)
        migrate_median = statistics.median(migrate_values)
        delta = abs(migrate_median - original_median)
        noise = max(
            max(original_values) - min(original_values),
            max(migrate_values) - min(migrate_values),
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

    with (result_root / "memory-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons


def resolve_memory_comparisons(rows, confirmation_rows):
    def key(row):
        return (
            row["collective"],
            row["algorithm"],
            row["compressor"],
            row["depth"],
            row["bytes"],
        )

    confirmations = {key(row): row for row in confirmation_rows}
    initial_failures = [
        row for row in rows if str(row["within_noise"]) != "1"
    ]
    final_failures = []
    confirmed_candidates = 0
    for row in initial_failures:
        confirmation = confirmations.get(key(row))
        if confirmation is not None:
            row = confirmation
            confirmed_candidates += 1
        if str(row["within_noise"]) != "1":
            final_failures.append(row)
    return initial_failures, confirmed_candidates, final_failures


def main():
    result_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/data/home/scyb672/run/lxc/COCCL-migrate/results/M0"
    )
    completeness = read_key_values(result_root / "completeness.txt")
    confirmation_path = result_root / "confirmation-comparison.csv"
    confirmation_rows = (
        read_csv(confirmation_path) if confirmation_path.exists() else []
    )
    performance = performance_summary(
        read_csv(result_root / "comparison.csv"), confirmation_rows
    )
    memory = write_memory_comparison(
        result_root, read_csv(result_root / "memory.csv")
    )
    memory_confirmation_path = (
        result_root / "memory-confirmation-comparison.csv"
    )
    memory_confirmation_rows = (
        read_csv(memory_confirmation_path)
        if memory_confirmation_path.exists()
        else []
    )
    (
        initial_memory_failures,
        confirmed_memory_candidates,
        memory_failures,
    ) = resolve_memory_comparisons(memory, memory_confirmation_rows)
    teardown_diagnostics = row_count(
        result_root / "process-teardown-failures.csv"
    )
    environment_retries = row_count(
        result_root / "process-environment-retries.csv"
    )

    performance_complete = all(
        completeness.get(f"performance_{variant}")
        == completeness.get("expected_performance_per_variant")
        for variant in ("original", "migrate-copy")
    ) and all(
        completeness.get(f"data_rows_{variant}")
        == completeness.get("expected_data_rows_per_variant")
        for variant in ("original", "migrate-copy")
    )
    memory_complete = all(
        completeness.get(f"memory_{variant}")
        == completeness.get("expected_memory_per_variant")
        for variant in ("original", "migrate-copy")
    )
    gate_passed = (
        performance_complete
        and memory_complete
        and performance is not None
        and performance["gate_failures"] == 0
        and not memory_failures
    )

    lines = [
        "# M0 Baseline Report",
        "",
        f"Status: {'PASS' if gate_passed else 'IN PROGRESS OR FAILED'}",
        "",
        "## Provenance",
        "",
        "- Baseline commit: b71f216f8bbf8837dc1a8f87bd3ac88ab0f822dc.",
        "- Original and migrate-copy use the same production sources.",
        "- The migrate-copy diff under src/ is empty.",
        "- The original worktree contains repository-wide executable-bit noise and test-harness edits; these are recorded under revisions/.",
        "- The current refine snapshot is not a Git worktree and is not part of the M0 gate.",
        "",
        "## Environment",
        "",
        "- Host: d1n41a14g01.",
        "- GPUs: 4 x NVIDIA A800-SXM4-80GB, NV8 between every pair.",
        "- CUDA: 12.4; NCCL: 2.21.5.",
        "- Both baseline libraries contain only sm_80 cubins.",
        "",
        "## Completeness",
        "",
        "| Item | Original | Migrate-copy | Expected |",
        "|---|---:|---:|---:|",
        (
            "| Performance processes | "
            f"{completeness.get('performance_original', '0')} | "
            f"{completeness.get('performance_migrate-copy', '0')} | "
            f"{completeness.get('expected_performance_per_variant', '0')} |"
        ),
        (
            "| Performance data rows | "
            f"{completeness.get('data_rows_original', '0')} | "
            f"{completeness.get('data_rows_migrate-copy', '0')} | "
            f"{completeness.get('expected_data_rows_per_variant', '0')} |"
        ),
        (
            "| Memory processes | "
            f"{completeness.get('memory_original', '0')} | "
            f"{completeness.get('memory_migrate-copy', '0')} | "
            f"{completeness.get('expected_memory_per_variant', '0')} |"
        ),
        "",
        "## Performance Gate",
        "",
    ]

    if performance is None:
        lines.append("No paired performance rows have been parsed.")
    else:
        lines.extend(
            [
                f"- Paired points: {performance['points']}.",
                (
                    "- Hard-gate points (message size >= 512 MiB): "
                    f"{performance['gate_points']}."
                ),
                (
                    "- Initial hard-gate candidates outside +/-3%: "
                    f"{performance['initial_gate_failures']}."
                ),
                (
                    "- Candidates with balanced two-run confirmation: "
                    f"{performance['confirmed_candidates']}."
                ),
                (
                    "- Final hard-gate failures after confirmation: "
                    f"{performance['gate_failures']}."
                ),
                (
                    "- Points below 512 MiB outside +/-3% "
                    "(informational): "
                    f"{performance['informational_failures']}."
                ),
                (
                    "- All-size median absolute relative delta: "
                    f"{performance['median_abs_delta'] * 100:.3f}%."
                ),
                (
                    "- Largest hard-gate regression: "
                    f"{performance['max_regression'] * 100:.3f}% at "
                    f"{performance['max_regression_case']}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Memory Gate",
            "",
            f"- Paired memory cases: {len(memory)}.",
            (
                "- Initial candidates outside measured noise: "
                f"{len(initial_memory_failures)}."
            ),
            (
                "- Candidates with balanced two-run confirmation: "
                f"{confirmed_memory_candidates}."
            ),
            (
                "- Final cases outside measured noise after confirmation: "
                f"{len(memory_failures)}."
            ),
            "- Every memory case and confirmation uses two isolated process samples; noise is the larger per-variant run range, with a 4 MiB floor for four integer-MiB GPU samples.",
            "",
            "## Build Notes",
            "",
            "- The M0 build harness uses the repository's normal make path; generate.py is invoked by the device Makefile.",
            "- TACO's compressor callback was aligned with the rank-aware function table and is built, but the canonical M0 performance script tests only sdp4bit and cuzfp.",
            "- ZFP is built offline with the node's cmake/3.31.12 module.",
            "- Retired ReduceScatter/AllReduce Ring experiments are excluded: the current migration target removed them, and the original AllReduce Ring in-place path reproducibly raises an illegal memory access.",
            "- ReduceScatter TwoShot and AllReduce TripleShot are not part of the single-node M0 script.",
            "- The test-only final cudaDeviceReset was removed from both baseline trees because it caused post-footer teardown crashes with COCCL's private multi-GPU streams.",
            f"- Diagnostic pre-fix teardown events retained: {teardown_diagnostics}; environment retries retained: {environment_retries}.",
            "",
            "## Correctness Scope",
            "",
            "All tests use -c 0. Completion and Out of bounds values: 0 are recorded, but this report does not claim numerical correctness.",
            "",
        ]
    )

    (result_root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
