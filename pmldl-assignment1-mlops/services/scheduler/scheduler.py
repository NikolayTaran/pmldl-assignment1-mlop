#!/usr/bin/env python3
"""Scheduler — runs the complete pipeline every 5 minutes.

    python services/scheduler/scheduler.py

Equivalent cron entry (see README):

    */5 * * * *  cd /path/to/pmldl-assignment1-mlops && /usr/bin/python3 services/scheduler/pipeline.py >> reports/pipeline.log 2>&1

Stop the scheduler with Ctrl+C. Each cycle logs its output to the console
(redirect it to a file to keep the history, as in the cron line above).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline every N minutes")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="minutes between pipeline runs (default: 5)")
    args = parser.parse_args()

    interval_seconds = args.interval * 60
    print(
        f"[scheduler] started — running the pipeline every {args.interval:g} minute(s). "
        "Press Ctrl+C to stop.",
        flush=True,
    )
    while True:
        started = time.time()
        try:
            run_pipeline()
        except Exception as exc:  # noqa: BLE001 — keep the scheduler alive
            print(f"[scheduler] pipeline run failed: {exc}", flush=True)
        elapsed = time.time() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        next_at = datetime.fromtimestamp(time.time() + sleep_for, tz=timezone.utc)
        print(
            f"[scheduler] cycle took {elapsed:.1f}s; next run at "
            f"{next_at.strftime('%H:%M:%S')} UTC (in {sleep_for / 60:.1f} min)",
            flush=True,
        )
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[scheduler] stopped by user")
