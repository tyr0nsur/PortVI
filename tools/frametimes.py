"""Summarize PresentMon display times for each process and swap chain."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(stream):
    reader = csv.DictReader(stream)
    required = {"Application", "ProcessID", "SwapChainAddress", "DisplayedTime"}
    missing = required.difference(reader.fieldnames or ())
    if missing:
        raise ValueError("Missing PresentMon columns: " + ", ".join(sorted(missing)))
    groups = defaultdict(lambda: {"times": [], "undisplayed": 0, "zero": 0, "types": Counter()})
    for row in reader:
        key = (row["Application"], row["ProcessID"], row["SwapChainAddress"])
        if not all(key):
            raise ValueError(f"Missing stream identity at CSV line {reader.line_num}")
        group = groups[key]
        value = (row["DisplayedTime"] or "").strip()
        if value.upper() == "NA":
            group["undisplayed"] += 1
            continue
        try:
            duration = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid DisplayedTime at CSV line {reader.line_num}: {value!r}") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"Invalid DisplayedTime at CSV line {reader.line_num}: {value!r}")
        if duration == 0:
            group["zero"] += 1
            continue
        group["times"].append(duration)
        group["types"][row.get("FrameType") or "unreported"] += 1
    if not groups:
        raise ValueError("The capture has no frame rows.")
    reports = []
    for (app, pid, chain), group in sorted(groups.items()):
        times = sorted(group["times"])
        report = dict(application=app, pid=pid, swap_chain=chain, displayed_intervals=len(times),
                      undisplayed_rows=group["undisplayed"], zero_duration_rows=group["zero"],
                      displayed_frame_types=dict(group["types"]))
        if times:
            def percentile(p):
                return times[math.ceil(len(times) * p / 100) - 1]

            report.update(measured_display_seconds=round(math.fsum(times) / 1000, 3),
                          displayed_fps=round(1000 / statistics.fmean(times), 3),
                          p95_ms=percentile(95), p99_ms=percentile(99), worst_ms=times[-1])
        reports.append(report)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        with args.capture.open(encoding="utf-8-sig", newline="") as capture:
            print(json.dumps(summarize(capture), indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as exc:
        sys.exit(str(exc))
