"""Run the whole study.

    python -m sps_validation.run_all --pilot    # prepare + 10 trial requests per judge + cost plan (cents)
    python -m sps_validation.run_all --judge    # prepare + sample sized to the budget + both judges + report

Options: --budget 16 (US$ per account, default 16), --fresh (redo the preparation).
The paid steps need ANTHROPIC_API_KEY and OPENAI_API_KEY. Before the full run
the estimated cost is shown and nothing is spent until you type yes. Judge
runs are resumable: an interrupted run picks up its submitted batches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PREP = [
    ["sps_validation.sources"],
    ["sps_validation.ingest"],
    ["sps_validation.segment"],
    ["sps_validation.features"],
    ["sps_validation.align"],
    ["sps_validation.judge", "prepare"],
]
SETS = ("main", "retest", "perturb")


def step(args: list[str]) -> None:
    print("==>", " ".join(args), flush=True)
    # UTF-8 everywhere (Windows consoles default to a legacy code page)
    subprocess.run([sys.executable, "-m", *args], check=True, env=dict(os.environ, PYTHONUTF8="1"))


def main(argv: list[str]) -> None:
    budget = argv[argv.index("--budget") + 1] if "--budget" in argv else "16"
    if "--fresh" in argv or not Path("build/judge/sets/all.jsonl").exists():
        for a in PREP:
            step(a)
    else:
        print("Preparation already done (use --fresh to redo it).")
    if "--judge" not in argv and "--pilot" not in argv:
        return
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing API keys: {', '.join(missing)}")
    if "--pilot" in argv:
        for judge in ("claude", "gpt"):
            step(["sps_validation.judge", "pilot", judge, "10"])
        step(["sps_validation.judge", "pilot-report"])
        step(["sps_validation.plan", "estimate", "--budget", budget])
        return
    step(["sps_validation.plan", "write", "--budget", budget])
    answer = input("\nStart the paid run with this plan? Type yes to continue: ").strip().lower()
    if answer != "yes":
        print("Stopped. Nothing was spent.")
        return
    for judge in ("claude", "gpt"):
        for s in SETS:
            step(["sps_validation.judge", "run", judge, s])
    step(["sps_validation.report"])
    print("\nDone. Results: build/report/report.md and build/report/sentences.csv")


if __name__ == "__main__":
    main(sys.argv)
