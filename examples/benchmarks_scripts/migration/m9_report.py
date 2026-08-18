#!/usr/bin/env python3

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GATE_BYTES = 512 << 20
EXPECTED_PERFORMANCE = {"native": 64, "sdp4bit": 238, "zfp": 238}
EXPECTED_SMOKE = {"native": 4, "sdp4bit": 15, "zfp": 5}
EXPECTED_MEMORY = {"sdp4bit": 13, "zfp": 12}
MILESTONES = {
    "alltoall": ("M5", "alltoall.csv"),
    "allgather": ("M6", "allgather.csv"),
    "reducescatter": ("M7", "reducescatter.csv"),
    "allreduce": ("M8", "allreduce.csv"),
}


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0],
        "algorithm": parts[1],
        "compressor": parts[2],
        "depth": int(parts[3][1:]),
        "bytes": int(parts[4][1:]),
    }


def parse_benchmark(path, case):
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            base = {**case, "bytes": int(tokens[0])}
            rows.append({
                **base, "inplace": 0, "time_us": float(tokens[5]),
                "algbw_gbps": float(tokens[6]),
                "busbw_gbps": float(tokens[7]),
            })
            rows.append({
                **base, "inplace": 1, "time_us": float(tokens[9]),
                "algbw_gbps": float(tokens[10]),
                "busbw_gbps": float(tokens[11]),
            })
    return rows


def write_performance(result_root):
    rows = []
    counts = {}
    for compressor in EXPECTED_PERFORMANCE:
        raw = result_root / compressor / "raw" / "performance"
        markers = sorted(raw.glob("*.ok"))
        counts[compressor] = len(markers)
        for marker in markers:
            rows.extend(parse_benchmark(marker.with_suffix(".log"),
                                        case_fields(marker)))
    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "time_us", "algbw_gbps", "busbw_gbps",
    ]
    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, counts


def row_key(row, compressor=None):
    return (
        row["collective"], row["algorithm"],
        compressor if compressor is not None else row["compressor"],
        int(row["depth"]), int(row["bytes"]), int(row["inplace"]),
    )


def milestone_rows(history_root):
    rows = {}
    for collective, (milestone, filename) in MILESTONES.items():
        with (history_root / milestone / filename).open() as stream:
            for row in csv.DictReader(stream):
                rows[row_key(row)] = float(row["time_us"])
    return rows


def m0_rows(history_root):
    grouped = defaultdict(list)
    with (history_root / "M0" / "summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != "original":
                continue
            grouped[row_key(row)].append(float(row["time_us"]))
    return {key: statistics.median(values)
            for key, values in grouped.items()}


def layout_rows(history_root):
    result = {}
    for collective, (milestone, _) in MILESTONES.items():
        with (history_root / milestone / "layout.csv").open() as stream:
            for row in csv.DictReader(stream):
                result[(collective, int(row["bytes"]),
                        int(row["depth"]))] = float(row["layout_us"])
    return result


def compare_performance(result_root, rows, history_root):
    current_baseline = milestone_rows(history_root)
    original_baseline = m0_rows(history_root)
    layouts = layout_rows(history_root)
    comparisons = []
    failures = []
    missing = []
    for row in rows:
        if row["compressor"] == "zfp":
            key = row_key(row, "cuzfp")
            old_us = original_baseline.get(key)
            layout_us = (layouts.get(
                (row["collective"], row["bytes"], row["depth"]), 0.0)
                if row["depth"] > 1 else 0.0)
            rule = "M0-cuzfp+layout+5%"
            limit_us = None if old_us is None else old_us * 1.05 + layout_us
        else:
            key = row_key(row)
            old_us = current_baseline.get(key)
            layout_us = 0.0
            factor = 1.03 if row["compressor"] == "native" else 1.05
            rule = "M8-native+3%" if factor == 1.03 else "M8-sdp4bit+5%"
            limit_us = None if old_us is None else old_us * factor

        gate = row["bytes"] >= GATE_BYTES and row["inplace"] == 0
        if old_us is None:
            if gate:
                missing.append(key)
            continue
        passed = not gate or row["time_us"] <= limit_us
        comparison = {
            **{name: row[name] for name in (
                "collective", "algorithm", "compressor", "depth",
                "bytes", "inplace")},
            "baseline_us": old_us,
            "m9_us": row["time_us"],
            "layout_us": layout_us,
            "limit_us": limit_us,
            "relative_delta": (row["time_us"] - old_us) / old_us,
            "gate": int(gate), "rule": rule, "passed": int(passed),
        }
        comparisons.append(comparison)
        if not passed:
            failures.append(comparison)

    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "inplace", "baseline_us", "m9_us", "layout_us", "limit_us",
        "relative_delta", "gate", "rule", "passed",
    ]
    with (result_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons, failures, missing


def write_crossover(result_root, rows):
    native = {
        (row["collective"], row["bytes"], row["inplace"]): row["time_us"]
        for row in rows if row["compressor"] == "native"
    }
    crossed = {}
    for row in rows:
        if row["compressor"] == "native" or row["inplace"] != 0:
            continue
        key = (row["compressor"], row["collective"], row["algorithm"],
               row["depth"])
        native_us = native.get((row["collective"], row["bytes"], 0))
        if native_us is not None and row["time_us"] <= native_us:
            crossed[key] = min(crossed.get(key, row["bytes"]), row["bytes"])
    fields = ["compressor", "collective", "algorithm", "depth",
              "crossover_bytes"]
    with (result_root / "crossover.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, size in sorted(crossed.items()):
            writer.writerow(dict(zip(fields, (*key, size))))
    return crossed


def peak_memory(path):
    peaks = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                peaks[fields[2]] = max(peaks[fields[2]], int(fields[3]))
    return sum(peaks.values()), len(peaks)


SCRATCH = re.compile(r"COCCL compressor (\S+) scratch device (\d+) peak (\d+)")
PERSISTENT = re.compile(
    r"COCCL compressor (\S+) persistent device (\d+) slot (\d+) bytes (\d+)")
COMPRESSOR_RELEASE = re.compile(
    r"COCCL compressor (\S+) release device (\d+) persistent (\d+) "
    r"scratch_peak (\d+) states (\d+)")
VMM_RELEASE = re.compile(
    r"COCCL VMM release .*remaining_virtual (\d+) "
    r"remaining_physical (\d+) remaining_registered (\d+)")


def workspace_rows(history_root):
    result = {}
    for collective, (milestone, _) in MILESTONES.items():
        with (history_root / milestone / "workspace-plan.csv").open() as stream:
            for row in csv.DictReader(stream):
                algorithm = row.get("algorithm")
                if not algorithm:
                    algorithm = {
                        "alltoall": "pipeline", "allgather": "pipeline",
                        "reducescatter": "oneshot",
                    }[collective]
                key = (collective, algorithm, int(row["bytes"]),
                       int(row["requested_depth"]))
                result[key] = max(result.get(key, 0),
                                  int(row["workspace_bytes"]))
    return result


def write_memory(result_root, history_root):
    workspaces = workspace_rows(history_root)
    rows = []
    release_failures = []
    counts = {}
    for compressor in EXPECTED_MEMORY:
        raw = result_root / compressor / "raw" / "memory"
        markers = sorted(raw.glob("*.ok"))
        counts[compressor] = len(markers)
        for marker in markers:
            case = case_fields(marker)
            log_text = marker.with_suffix(".log").read_text(errors="replace")
            sample = (result_root / compressor / "memory-samples" /
                      (marker.stem + ".csv"))
            peak, gpu_count = peak_memory(sample)
            scratch = {(name, int(device)): int(size)
                       for name, device, size in SCRATCH.findall(log_text)}
            persistent = {(name, int(device), int(slot)): int(size)
                          for name, device, slot, size
                          in PERSISTENT.findall(log_text)}
            releases = COMPRESSOR_RELEASE.findall(log_text)
            vmm_releases = VMM_RELEASE.findall(log_text)
            vmm_ok = len(vmm_releases) >= 4 and all(
                all(int(value) == 0 for value in release)
                for release in vmm_releases)
            subadd = "__profilesubadd" in marker.stem
            compressor_release_ok = not subadd or (
                len(releases) >= 4 and all(
                    int(values[2]) > 0 and int(values[4]) > 0
                    for values in releases if values[0] == "sdp4bit"))
            if not vmm_ok or not compressor_release_ok:
                release_failures.append(marker.stem)
            key = (case["collective"], case["algorithm"], case["bytes"],
                   case["depth"])
            rows.append({
                **case,
                "peak_total_mib": peak,
                "sampled_gpus": gpu_count,
                "workspace_bytes_per_rank": workspaces.get(key, 0),
                "scratch_peak_total_bytes": sum(scratch.values()),
                "persistent_total_bytes": sum(persistent.values()),
                "compressor_release_ok": int(compressor_release_ok),
                "vmm_release_ok": int(vmm_ok),
            })
    fields = [
        "collective", "algorithm", "compressor", "depth", "bytes",
        "peak_total_mib", "sampled_gpus", "workspace_bytes_per_rank",
        "scratch_peak_total_bytes", "persistent_total_bytes",
        "compressor_release_ok", "vmm_release_ok",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, counts, release_failures


def write_parameters(result_root):
    (result_root / "plugin-parameters.md").write_text(
        "# M9 Plugin Parameter Mapping\n\n"
        "| M9 plugin | M0 label | A2A | AG | RS | AR |\n"
        "|---|---|---|---|---|---|\n"
        "| sdp4bit | sdp4bit | 4-bit, group 2048 | 4-bit, group 2048 | "
        "4-bit, group 128, Hadamard | 4-bit, group 128 |\n"
        "| zfp | cuzfp | rate 4 | rate 8 | rate 8 | rate 8 |\n\n"
        "TahQuant and TACO are build/load-only in M9.\n"
    )


def main():
    result_root = Path(sys.argv[1])
    history_root = Path(sys.argv[2])
    rows, performance_counts = write_performance(result_root)
    comparisons, gate_failures, missing_baselines = compare_performance(
        result_root, rows, history_root)
    crossovers = write_crossover(result_root, rows)
    memory_rows, memory_counts, release_failures = write_memory(
        result_root, history_root)
    write_parameters(result_root)

    smoke_counts = {
        name: len(list((result_root / name / "raw" / "smoke").glob("*.ok")))
        for name in EXPECTED_SMOKE
    }
    one_shot_over_limit = [
        row for row in rows
        if row["collective"] == "allreduce" and
        row["algorithm"] == "oneshot" and row["bytes"] > 32 << 20
    ]
    host_ok = "passed" in (
        result_root / "host-fixed-compressors.txt").read_text(errors="replace")
    layout_ok = "PASS" in (
        result_root / "layout-correctness.txt").read_text(errors="replace")
    memory_gpu_ok = bool(memory_rows) and all(
        row["sampled_gpus"] == 4 for row in memory_rows)
    complete = (
        performance_counts == EXPECTED_PERFORMANCE and
        smoke_counts == EXPECTED_SMOKE and memory_counts == EXPECTED_MEMORY and
        len(rows) == 2 * sum(EXPECTED_PERFORMANCE.values()))
    passed = (complete and host_ok and layout_ok and memory_gpu_ok and
              not gate_failures and not missing_baselines and
              not release_failures and not one_shot_over_limit)

    gated = [row for row in comparisons if row["gate"]]
    sdp_reduction = [
        row for row in gated
        if row["compressor"] == "sdp4bit" and
        row["collective"] in ("reducescatter", "allreduce")
    ]
    sdp_reduction_median = (statistics.median(
        row["relative_delta"] for row in sdp_reduction)
        if sdp_reduction else None)
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M9 Fixed Compressor Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write("## Completeness\n\n")
        stream.write(f"- Performance processes: {performance_counts} / {EXPECTED_PERFORMANCE}.\n")
        stream.write(f"- Smoke processes: {smoke_counts} / {EXPECTED_SMOKE}.\n")
        stream.write(f"- Memory processes: {memory_counts} / {EXPECTED_MEMORY}.\n")
        stream.write(f"- Parsed performance rows: {len(rows)}.\n")
        stream.write(f"- Host ABI/config/layout test: {'PASS' if host_ok else 'FAIL'}.\n\n")
        stream.write(f"- GPU Pack/Unpack layout test: {'PASS' if layout_ok else 'FAIL'}.\n\n")
        stream.write("## Performance\n\n")
        stream.write(f"- Gated out-of-place points at >=512 MiB: {len(gated)}.\n")
        stream.write(f"- Gate failures: {len(gate_failures)}.\n")
        stream.write(f"- Missing gated baselines: {len(missing_baselines)}.\n")
        stream.write(f"- Reported crossover groups: {len(crossovers)}.\n")
        stream.write("- SDP4Bit uses the matching M5-M8 result plus 5%.\n")
        stream.write("- ZFP is reported as M0 `cuzfp` plus Pack/Unpack and 5%.\n")
        stream.write("- Native uses the matching M5-M8 result plus 3%.\n")
        stream.write("- AllReduce OneShot is limited to 4 KiB..32 MiB.\n\n")
        stream.write("## SDP4Bit 2D Block\n\n")
        if sdp_reduction_median is None:
            stream.write("- No completed >=512 MiB reduction comparison.\n\n")
        else:
            stream.write(
                f"- RS OneShot/AR TwoShot median delta versus M7/M8: "
                f"{sdp_reduction_median:+.2%} across {len(sdp_reduction)} points.\n")
            stream.write(
                "- Per-size/depth values are retained in `comparison.csv`; "
                "negative values are improvements.\n\n")
        stream.write("## Memory And Lifetime\n\n")
        stream.write(f"- Four-GPU samples complete: {'PASS' if memory_gpu_ok else 'FAIL'}.\n")
        stream.write(f"- Resource release failures: {len(release_failures)}.\n")
        stream.write("- `memory.csv` separates planner workspace, compressor scratch, persistent state, and process peak.\n")
        stream.write("- TahQuant and TACO are compile/load-only; see `build-manifest.txt`.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
