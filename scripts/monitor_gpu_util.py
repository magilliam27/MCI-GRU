#!/usr/bin/env python
"""Sample ``nvidia-smi`` GPU telemetry to a CSV file until stopped."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

QUERY_FIELDS = [
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Optional sentinel path. Sampling stops when this file exists.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Write one sample and exit.",
    )
    return parser.parse_args()


def read_gpu_row() -> list[str]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=True)
    line = proc.stdout.strip().splitlines()[0]
    return [part.strip() for part in line.split(",")]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path = Path(args.stop_file) if args.stop_file else None
    fieldnames = [
        "timestamp_utc",
        "gpu_name",
        "utilization_gpu_pct",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            try:
                name, util, mem_used, mem_total, power = read_gpu_row()
                writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "gpu_name": name,
                        "utilization_gpu_pct": util,
                        "memory_used_mib": mem_used,
                        "memory_total_mib": mem_total,
                        "power_draw_w": power,
                    }
                )
                handle.flush()
            except Exception as exc:
                writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "gpu_name": "ERROR",
                        "utilization_gpu_pct": "",
                        "memory_used_mib": "",
                        "memory_total_mib": "",
                        "power_draw_w": str(exc),
                    }
                )
                handle.flush()

            if args.once or (stop_path is not None and stop_path.exists()):
                break
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    main()
