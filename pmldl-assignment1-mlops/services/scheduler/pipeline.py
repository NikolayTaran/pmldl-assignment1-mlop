#!/usr/bin/env python3
"""Pipeline orchestrator — connects all three stages.

    Stage 1  data engineering    python code/datasets/process_data.py
    Stage 2  model engineering   python code/models/train_model.py
    Stage 3  deployment          docker compose up -d --build

Run once:
    python services/scheduler/pipeline.py

Run every 5 minutes (see scheduler.py, cron or systemd in the README):
    python services/scheduler/scheduler.py

The pipeline is idempotent: every cycle overwrites data/processed,
models/model.pkl and restarts the containers with the fresh model.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "code" / "deployment" / "docker-compose.yml"

STAGES = [
    ("stage1-data-engineering", [sys.executable, "code/datasets/process_data.py"]),
    ("stage2-model-engineering", [sys.executable, "code/models/train_model.py"]),
]


def log(stage: str, message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{stage}] {message}", flush=True)


def run_stage(name: str, cmd: list[str]) -> None:
    log(name, f"running: {' '.join(cmd)}")
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode != 0:
        log(name, f"FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr.rstrip(), flush=True)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    log(name, f"finished in {time.time() - started:.1f}s")


def deploy() -> None:
    """Stage 3 — rebuild and (re)start the api + app containers."""
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"]
    log("stage3-deployment", f"running: {' '.join(cmd)}")
    started = time.time()
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "docker is not installed / not on PATH. "
            "Install Docker Desktop (https://www.docker.com/products/docker-desktop/) "
            "and make sure 'docker compose version' works, then re-run the pipeline."
        ) from None
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode != 0:
        log("stage3-deployment", f"FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr.rstrip(), flush=True)
        raise RuntimeError(f"deployment failed with exit code {result.returncode}")
    log("stage3-deployment", f"containers are up in {time.time() - started:.1f}s")


def run_pipeline() -> None:
    started = time.time()
    log("pipeline", "=== pipeline run started ===")
    for name, cmd in STAGES:
        run_stage(name, cmd)
    deploy()
    log("pipeline", f"=== pipeline run finished in {time.time() - started:.1f}s ===")


if __name__ == "__main__":
    run_pipeline()
