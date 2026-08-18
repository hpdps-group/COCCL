#!/usr/bin/env python3

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

PERFORMANCE_PROFILES = ("native", "fallback", "sdp4bit", "zfp", "zfp-raw")
MEMORY_PROFILES = PERFORMANCE_PROFILES
SMOKE_PROFILES = PERFORMANCE_PROFILES + ("auto-sdp4bit",)
DEADLOCK_PROFILES = ("auto-sdp4bit", "sdp4bit", "zfp", "zfp-raw")
PATTERNS = ("ring", "bidirectional-same-stream",
            "bidirectional-multistream")
GATE_BYTES = 512 << 20


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "pattern": parts[1],
        "profile": parts[2],
        "bytes": int(parts[3][1:]),
    }


def parse_benchmark(path, case):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) != 13 or not tokens[0].isdigit():
            continue
        rows.append({
            **case, "inplace": 0, "time_us": float(tokens[5]),
            "algbw_gbps": float(tokens[6]), "busbw_gbps": float(tokens[7]),
        })
        rows.append({
            **case, "inplace": 1, "time_us": float(tokens[9]),
            "algbw_gbps": float(tokens[10]),
            "busbw_gbps": float(tokens[11]),
        })
    return rows


def markers(result_root, profile, mode):
    return sorted((result_root / profile / "raw" / mode).glob("*.ok"))


def write_performance(result_root):
    rows = []
    counts = {}
    for profile in PERFORMANCE_PROFILES:
        current = markers(result_root, profile, "performance")
        counts[profile] = len(current)
        for marker in current:
            rows.extend(parse_benchmark(marker.with_suffix(".log"),
                                        case_fields(marker)))
    fields = [
        "pattern", "profile", "bytes", "inplace", "time_us",
        "algbw_gbps", "busbw_gbps",
    ]
    with (result_root / "sendrecv.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, counts


def fallback_gate(result_root, rows):
    indexed = {
        (row["profile"], row["bytes"], row["inplace"]): row
        for row in rows
    }
    comparisons = []
    failures = []
    for key, fallback in indexed.items():
        profile, size, inplace = key
        if profile != "fallback" or size < GATE_BYTES:
            continue
        native = indexed.get(("native", size, inplace))
        passed = native is not None and fallback["time_us"] <= (
            1.03 * native["time_us"])
        row = {
            "bytes": size, "inplace": inplace,
            "native_us": None if native is None else native["time_us"],
            "fallback_us": fallback["time_us"],
            "limit_us": None if native is None else 1.03 * native["time_us"],
            "passed": int(passed),
        }
        comparisons.append(row)
        if not passed:
            failures.append(row)
    fields = [
        "bytes", "inplace", "native_us", "fallback_us", "limit_us",
        "passed",
    ]
    with (result_root / "native_fallback_comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons, failures


def peak_memory(path):
    peaks = defaultdict(int)
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            peaks[fields[2]] = max(peaks[fields[2]], int(fields[3]))
    return sum(peaks.values()), len(peaks)


VMM_RELEASE = re.compile(
    r"COCCL VMM release .*remaining_virtual (\d+) "
    r"remaining_physical (\d+) remaining_registered (\d+)")


def write_memory(result_root):
    rows = []
    counts = {}
    failures = []
    for profile in MEMORY_PROFILES:
        current = markers(result_root, profile, "memory")
        counts[profile] = len(current)
        for marker in current:
            case = case_fields(marker)
            sample = result_root / profile / "memory-samples" / (
                marker.stem + ".csv")
            peak, gpu_count = peak_memory(sample)
            log = marker.with_suffix(".log").read_text(errors="replace")
            releases = VMM_RELEASE.findall(log)
            needs_release = profile not in ("native", "fallback")
            release_ok = not needs_release or (
                len(releases) >= 4 and all(
                    all(int(value) == 0 for value in entry)
                    for entry in releases))
            row = {
                **case, "peak_total_mib": peak, "sampled_gpus": gpu_count,
                "vmm_release_ok": int(release_ok),
            }
            rows.append(row)
            if gpu_count != 4 or not release_ok:
                failures.append(row)
    fields = [
        "pattern", "profile", "bytes", "peak_total_mib",
        "sampled_gpus", "vmm_release_ok",
    ]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, counts, failures


def write_deadlocks(result_root):
    rows = []
    complete = True
    for profile in DEADLOCK_PROFILES:
        current = markers(result_root, profile, "deadlock")
        by_pattern = {case_fields(marker)["pattern"]: marker
                      for marker in current}
        for pattern in PATTERNS:
            marker = by_pattern.get(pattern)
            passed = marker is not None
            complete &= passed
            time_us = None
            if marker is not None:
                parsed = parse_benchmark(marker.with_suffix(".log"),
                                         case_fields(marker))
                time_us = parsed[0]["time_us"] if parsed else None
                passed = bool(parsed)
            rows.append((profile, pattern, passed, time_us))
            complete &= passed
    with (result_root / "deadlock_cases.md").open("w") as stream:
        stream.write("# M13 Grouped Send/Recv Deadlock Cases\n\n")
        stream.write("| Profile | Pattern | Result | First time (us) |\n")
        stream.write("|---|---|---|---:|\n")
        for profile, pattern, passed, time_us in rows:
            shown = "" if time_us is None else f"{time_us:.1f}"
            stream.write(
                f"| {profile} | {pattern} | "
                f"{'PASS' if passed else 'FAIL'} | {shown} |\n")
        stream.write("\nEvery case runs in an external timeout. A timeout is a deadlock failure.\n")
    return rows, complete


def main():
    result_root = Path(sys.argv[1])
    rows, performance_counts = write_performance(result_root)
    comparisons, fallback_failures = fallback_gate(result_root, rows)
    _, memory_counts, memory_failures = write_memory(result_root)
    _, deadlocks_ok = write_deadlocks(result_root)
    smoke_counts = {
        profile: len(markers(result_root, profile, "smoke"))
        for profile in SMOKE_PROFILES
    }
    complete = (
        performance_counts == {profile: 5 for profile in PERFORMANCE_PROFILES}
        and memory_counts == {profile: 5 for profile in MEMORY_PROFILES}
        and smoke_counts == {profile: 1 for profile in SMOKE_PROFILES}
        and len(rows) == 2 * 5 * len(PERFORMANCE_PROFILES)
        and len(comparisons) == 6
        and deadlocks_ok
    )
    passed = complete and not fallback_failures and not memory_failures
    with (result_root / "report.md").open("w") as stream:
        stream.write("# M13 Grouped Byte-protocol Send/Recv Report\n\n")
        stream.write(f"Status: {'PASS' if passed else 'FAIL'}\n\n")
        stream.write(f"- Performance processes: {performance_counts}.\n")
        stream.write(f"- Smoke processes: {smoke_counts}.\n")
        stream.write(f"- Memory processes: {memory_counts}.\n")
        stream.write(
            f"- Native fallback regressions above 3% at >=512 MiB: "
            f"{len(fallback_failures)}.\n")
        stream.write(
            f"- Memory sampling or communicator release failures: "
            f"{len(memory_failures)}.\n")
        stream.write(
            "- SDP4Bit and ZFP timings are informational; the hard gate is "
            "completion without deadlock.\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
