"""Run the whole study.

    python -m sps_validation.run_all            # free steps only (data -> requests)
    python -m sps_validation.run_all --pilot    # free steps + 20 trial requests per judge (a few dollars)
    python -m sps_validation.run_all --judge    # free steps + both judges in full (paid) + the report

The paid steps need ANTHROPIC_API_KEY and OPENAI_API_KEY. Each judge run is
resumable: an interrupted run picks up its submitted batches.
"""

from __future__ import annotations

import os
import subprocess
import sys

FREE = [
    ["sps_validation.sources"],
    ["sps_validation.ingest"],
    ["sps_validation.segment"],
    ["sps_validation.features"],
    ["sps_validation.align"],
    ["sps_validation.judge", "prepare"],
    ["sps_validation.validate", "prepare"],
]
PAID = [
    ["sps_validation.judge", "run", judge, s]
    for judge in ("claude", "gpt")
    for s in ("main", "retest", "perturb")
]


def step(args: list[str]) -> None:
    print("==>", " ".join(args), flush=True)
    # UTF-8 everywhere (Windows consoles default to a legacy code page)
    subprocess.run([sys.executable, "-m", *args], check=True, env=dict(os.environ, PYTHONUTF8="1"))


def main(argv: list[str]) -> None:
    for a in FREE:
        step(a)
    if "--judge" not in argv and "--pilot" not in argv:
        print("Free steps done. Run with --pilot (trial) or --judge (full, paid).")
        return
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing API keys: {', '.join(missing)}")
    if "--pilot" in argv:
        for judge in ("claude", "gpt"):
            step(["sps_validation.judge", "pilot", judge, "20"])
        step(["sps_validation.judge", "pilot-report"])
        return
    for a in PAID:
        step(a)
    step(["sps_validation.report"])


if __name__ == "__main__":
    main(sys.argv)
