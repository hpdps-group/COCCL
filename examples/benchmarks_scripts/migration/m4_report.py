#!/usr/bin/env python3

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


PERFORMANCE_SIZES = (1 << 26, 1 << 29, 1 << 30, 1 << 33)
MEMORY_SIZES = (1 << 26, 1 << 30, 1 << 33)
GATE_MIN_BYTES = 1 << 29
PHYSICAL_CHUNK_BYTES = 8 << 20


def case_fields(path):
    parts = path.stem.split("__")
    return {
        "collective": parts[0],
        "algorithm": parts[1],
        "compressor": parts[2],
        "backend": parts[3],
        "bytes": int(parts[4][1:]),
    }


def parse_log(path):
    case = case_fields(path)
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            base = {**case, "bytes": int(tokens[0])}
            rows.extend([
                {**base, "inplace": 0, "time_us": float(tokens[5]),
                 "algbw_gbps": float(tokens[6]),
                 "busbw_gbps": float(tokens[7])},
                {**base, "inplace": 1, "time_us": float(tokens[9]),
                 "algbw_gbps": float(tokens[10]),
                 "busbw_gbps": float(tokens[11])},
            ])
    return rows


def parse_m3_control_log(path):
    parts = path.stem.split("__")
    case = {
        "compressor": parts[2],
        "bytes": int(parts[3][1:]),
    }
    rows = []
    with path.open(errors="replace") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 13 or not tokens[0].isdigit():
                continue
            rows.extend([
                {**case, "bytes": int(tokens[0]), "inplace": 0,
                 "time_us": float(tokens[5])},
                {**case, "bytes": int(tokens[0]), "inplace": 1,
                 "time_us": float(tokens[9])},
            ])
    return rows


def load_m3_control(result_root):
    markers = sorted((result_root / "raw" / "performance").glob("*.ok"))
    rows = []
    for marker in markers:
        rows.extend(parse_m3_control_log(marker.with_suffix(".log")))
    values = {
        (row["compressor"], row["bytes"], row["inplace"]): row["time_us"]
        for row in rows
    }
    return len(markers), rows, values


def load_m3_performance(path):
    values = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != "alltoall":
                continue
            key = (row["compressor"], int(row["bytes"]),
                   int(row["inplace"]))
            values[key] = float(row["time_us"])
    return values


def write_performance(result_root, m3_root, control_root):
    marker_root = result_root / "raw" / "performance"
    markers = sorted(marker_root.glob("*.ok"))
    rows = []
    for marker in markers:
        rows.extend(parse_log(marker.with_suffix(".log")))

    fields = ["collective", "algorithm", "compressor", "backend", "bytes",
              "inplace", "time_us", "algbw_gbps", "busbw_gbps"]
    with (result_root / "performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    historical = load_m3_performance(m3_root / "performance.csv")
    control_count, control_rows, control = load_m3_control(control_root)
    comparisons = []
    for row in rows:
        key = (row["compressor"], row["bytes"], row["inplace"])
        reference = control.get(key)
        recorded = historical.get(key)
        if reference is None or recorded is None:
            continue
        delta = (row["time_us"] - reference) / reference
        gate = row["bytes"] >= GATE_MIN_BYTES
        comparisons.append({
            "compressor": row["compressor"],
            "bytes": row["bytes"],
            "inplace": row["inplace"],
            "m3_recorded_us": recorded,
            "m3_control_us": reference,
            "m4_vmm_us": row["time_us"],
            "recorded_delta": (row["time_us"] - recorded) / recorded,
            "relative_delta": delta,
            "is_gate_point": int(gate),
            "passed": int(not gate or delta <= 0.03),
        })

    fields = ["compressor", "bytes", "inplace", "m3_recorded_us",
              "m3_control_us", "m4_vmm_us", "recorded_delta",
              "relative_delta", "is_gate_point", "passed"]
    with (result_root / "performance-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return len(markers), rows, comparisons, control_count, control_rows


def verify_smoke(result_root):
    marker_root = result_root / "raw" / "smoke"
    markers = sorted(marker_root.glob("*.ok"))
    rows = []
    backend_matches = 0
    for marker in markers:
        log = marker.with_suffix(".log")
        rows.extend(parse_log(log))
        expected = case_fields(marker)["backend"]
        label = "VMM" if expected == "vmm" else "legacy"
        if f"backend {label}" in log.read_text(errors="replace"):
            backend_matches += 1
    return len(markers), rows, backend_matches


def parse_memory_sample(path):
    peak_by_gpu = defaultdict(int)
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                peak_by_gpu[fields[2]] = max(peak_by_gpu[fields[2]],
                                             int(fields[3]))
    return sum(peak_by_gpu.values()), len(peak_by_gpu)


def load_m3_memory(path):
    values = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != "alltoall":
                continue
            values[(row["compressor"], int(row["bytes"]))] = int(
                row["peak_total_mib"])
    return values


def write_memory(result_root, m3_root):
    rows = []
    marker_root = result_root / "raw" / "memory"
    for sample in sorted((result_root / "memory-samples").glob("*.csv")):
        if not (marker_root / (sample.stem + ".ok")).exists():
            continue
        peak, gpus = parse_memory_sample(sample)
        rows.append({**case_fields(sample), "peak_total_mib": peak,
                     "sampled_gpus": gpus})

    fields = ["collective", "algorithm", "compressor", "backend", "bytes",
              "peak_total_mib", "sampled_gpus"]
    with (result_root / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    baseline = load_m3_memory(m3_root / "memory.csv")
    comparisons = []
    for row in rows:
        reference = baseline.get((row["compressor"], row["bytes"]))
        if reference is None:
            continue
        comparisons.append({
            "compressor": row["compressor"],
            "bytes": row["bytes"],
            "m3_legacy_peak_mib": reference,
            "m4_vmm_peak_mib": row["peak_total_mib"],
            "delta_mib": row["peak_total_mib"] - reference,
        })

    fields = ["compressor", "bytes", "m3_legacy_peak_mib",
              "m4_vmm_peak_mib", "delta_mib"]
    with (result_root / "memory-comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    return len(list(marker_root.glob("*.ok"))), rows, comparisons


def write_vmm_allocations(result_root):
    allocation_pattern = re.compile(
        r"COCCL VMM (allocation|growth) comm (0x[0-9a-fA-F]+) "
        r"requested ([0-9]+) reserved ([0-9]+) virtual ([0-9]+) "
        r"physical ([0-9]+) registered ([0-9]+)")
    registration_pattern = re.compile(
        r"COCCL VMM registration comm (0x[0-9a-fA-F]+) bytes "
        r"([0-9]+) total ([0-9]+)")
    release_pattern = re.compile(
        r"COCCL VMM release comm (0x[0-9a-fA-F]+) virtual ([0-9]+) "
        r"physical ([0-9]+) registered ([0-9]+) remaining_virtual "
        r"([0-9]+) remaining_physical ([0-9]+) remaining_registered "
        r"([0-9]+)")
    states = defaultdict(lambda: {
        "allocation_events": 0,
        "requested_bytes": 0,
        "virtual_reserved_bytes": 0,
        "physical_committed_bytes": 0,
        "registered_bytes": 0,
        "released_virtual_bytes": -1,
        "released_physical_bytes": -1,
        "released_registered_bytes": -1,
        "remaining_virtual_bytes": -1,
        "remaining_physical_bytes": -1,
        "remaining_registered_bytes": -1,
    })

    lifecycle_root = result_root / "raw" / "lifecycle"
    markers = sorted(lifecycle_root.glob("*.ok"))
    for marker in markers:
        with marker.with_suffix(".log").open(errors="replace") as stream:
            for line in stream:
                match = allocation_pattern.search(line)
                if match:
                    state = states[match.group(2)]
                    state["allocation_events"] += 1
                    state["requested_bytes"] = max(
                        state["requested_bytes"], int(match.group(3)))
                    state["virtual_reserved_bytes"] = max(
                        state["virtual_reserved_bytes"], int(match.group(5)))
                    state["physical_committed_bytes"] = max(
                        state["physical_committed_bytes"], int(match.group(6)))
                    state["registered_bytes"] = max(
                        state["registered_bytes"], int(match.group(7)))
                    continue
                match = registration_pattern.search(line)
                if match:
                    states[match.group(1)]["registered_bytes"] = max(
                        states[match.group(1)]["registered_bytes"],
                        int(match.group(3)))
                    continue
                match = release_pattern.search(line)
                if match:
                    state = states[match.group(1)]
                    state["released_virtual_bytes"] = int(match.group(2))
                    state["released_physical_bytes"] = int(match.group(3))
                    state["released_registered_bytes"] = int(match.group(4))
                    state["remaining_virtual_bytes"] = int(match.group(5))
                    state["remaining_physical_bytes"] = int(match.group(6))
                    state["remaining_registered_bytes"] = int(match.group(7))

    rows = []
    for comm, state in sorted(states.items()):
        delta = (state["physical_committed_bytes"] -
                 state["requested_bytes"])
        rows.append({
            "comm": comm,
            **state,
            "physical_request_delta_bytes": delta,
            "within_one_chunk": int(0 <= delta < PHYSICAL_CHUNK_BYTES),
            "released_all": int(
                state["remaining_virtual_bytes"] == 0 and
                state["remaining_physical_bytes"] == 0 and
                state["remaining_registered_bytes"] == 0),
        })

    fields = ["comm", "allocation_events", "requested_bytes",
              "virtual_reserved_bytes", "physical_committed_bytes",
              "registered_bytes", "physical_request_delta_bytes",
              "within_one_chunk", "released_virtual_bytes",
              "released_physical_bytes", "released_registered_bytes",
              "remaining_virtual_bytes", "remaining_physical_bytes",
              "remaining_registered_bytes", "released_all"]
    with (result_root / "vmm_allocations.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(markers), rows


def main():
    result_root = Path(sys.argv[1])
    m3_root = Path(sys.argv[2])
    control_root = (Path(sys.argv[3]) if len(sys.argv) > 3
                    else result_root / "m3-control")
    smoke_count, smoke_rows, smoke_backends = verify_smoke(result_root)
    (performance_count, performance_rows, comparisons, control_count,
     control_rows) = write_performance(result_root, m3_root, control_root)
    memory_count, memory_rows, memory_comparisons = write_memory(
        result_root, m3_root)
    lifecycle_count, allocation_rows = write_vmm_allocations(result_root)

    expected_performance = len(PERFORMANCE_SIZES) * 3
    expected_memory = len(MEMORY_SIZES) * 3
    gate_rows = [row for row in comparisons if row["is_gate_point"]]
    failures = [row for row in gate_rows if not row["passed"]]
    allocation_passed = (
        len(allocation_rows) == 4 and
        all(row["within_one_chunk"] and row["released_all"] and
            row["virtual_reserved_bytes"] >= row["requested_bytes"] and
            row["registered_bytes"] >= row["requested_bytes"]
            for row in allocation_rows)
    )
    host_passed = "M4 VMM buffer tests passed" in (
        result_root / "host-vmm.txt").read_text(errors="replace")
    complete = (
        host_passed and
        smoke_count == 6 and len(smoke_rows) == 12 and
        smoke_backends == 6 and
        performance_count == expected_performance and
        len(performance_rows) == expected_performance * 2 and
        len(comparisons) == expected_performance * 2 and
        control_count == expected_performance and
        len(control_rows) == expected_performance * 2 and
        memory_count == expected_memory and
        len(memory_rows) == expected_memory and
        len(memory_comparisons) == expected_memory and
        all(row["sampled_gpus"] == 4 for row in memory_rows) and
        lifecycle_count == 1 and allocation_passed
    )
    status = "PASS" if complete and not failures else "INCOMPLETE OR FAILED"

    lines = [
        "# M4 VMM Buffer Backend Report", "",
        f"Status: {status}", "", "## Scope", "",
        "- Added a VMM backend behind the unchanged M3 buffer-manager API.",
        "- Workspace request formulas, AllToAll execution, SDP4Bit, ZFP, and pipeline depth remain unchanged.",
        "- Legacy and VMM were smoke-tested explicitly; the full VMM matrix is gated against a same-time control built from frozen M3 commit f93a95f.",
        "- The original M3 measurements remain in the comparison CSV to expose temporal benchmark drift instead of hiding it.",
        "", "## Completeness", "",
        f"- Host lifecycle test: {'PASS' if host_passed else 'FAIL'}.",
        f"- Legacy/VMM smoke processes: {smoke_count}/6.",
        f"- Performance processes: {performance_count}/{expected_performance}.",
        f"- Performance rows: {len(performance_rows)}/{expected_performance * 2}.",
        f"- Frozen M3 control processes: {control_count}/{expected_performance}.",
        f"- Memory processes: {memory_count}/{expected_memory}.",
        f"- VMM lifecycle processes: {lifecycle_count}/1.",
        "", "## Performance Gate", "",
        f"- Gate rows at 512 MiB and above: {len(gate_rows)}.",
        f"- Failed rows: {len(failures)}; maximum regression is 3% against the same-time frozen M3 legacy control.",
        "- 64 MiB is informational.",
        "", "## Memory", "",
        "| Compressor | Bytes | M3 legacy MiB | M4 VMM MiB | Delta MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in memory_comparisons:
        lines.append(
            f"| {row['compressor']} | {row['bytes']} | "
            f"{row['m3_legacy_peak_mib']} | {row['m4_vmm_peak_mib']} | "
            f"{row['delta_mib']} |"
        )
    lines.extend([
        "", "## VMM Allocation", "",
        "| Comm | Events | Requested | Virtual | Physical | Registered | Delta | Released |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in allocation_rows:
        lines.append(
            f"| {row['comm']} | {row['allocation_events']} | "
            f"{row['requested_bytes']} | {row['virtual_reserved_bytes']} | "
            f"{row['physical_committed_bytes']} | {row['registered_bytes']} | "
            f"{row['physical_request_delta_bytes']} | {row['released_all']} |"
        )
    lines.extend([
        "", "## Verification", "",
        "- Host tests cover 8 MiB physical chunks, same-stream reuse, incremental growth, communicator isolation, release retry, backend parity, and teardown ordering.",
        "- Lifecycle logs record requested, virtual, physical, registered, and post-destroy remaining bytes for every rank.",
        "- Memory samples use one independent process per case and four GPUs.",
    ])
    if failures:
        lines.extend(["", "## Gate Failures", ""])
        for row in failures:
            lines.append(
                f"- {row['compressor']} {row['bytes']} inplace={row['inplace']}: "
                f"{row['relative_delta'] * 100:.2f}%."
            )
    (result_root / "report.md").write_text("\n".join(lines) + "\n")

    if not complete:
        raise SystemExit("M4 result matrix or VMM lifecycle evidence is incomplete")
    if failures:
        raise SystemExit("M4 performance gate failed")


if __name__ == "__main__":
    main()
