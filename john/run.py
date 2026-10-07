"""Run SATE on John 1: prepare, show the cost, judge (after you type yes), report.

As in the main study, three sets are judged: main, planted errors (+ controls) and a 10% retest.

The judges are those of the main study (sps_validation.judge: the same models, instructions,
schema, settings and Batch APIs). Only the folder changes: build/john/judge/.

    python -m john.run                       # both judges (Claude Haiku 4.5 and GPT-5 mini)
    python -m john.run --judges gpt          # one judge only
    python -m john.run --skip-prepare        # reuse build/john/ (e.g. after an interruption)
    python -m john.run --mock                # pipeline test, no API calls

The run can be interrupted: running the same command again resumes the submitted batches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sps_validation import judge as J

from . import prepare, report

SETS = ("main", "perturb", "retest")  # as in the main study

# Used only for the estimate when the main study's own results (build/judge/results/) are absent:
# tokens per character of prompt, output tokens per information item (GPT includes reasoning).
DEFAULT_RATE = {"claude": {"in_per_char": 1 / 3.2, "out_per_item": 90},
                "gpt": {"in_per_char": 1 / 3.2, "out_per_item": 260}}


def measured_rate(judge: str) -> dict | None:
    """Tokens actually used per prompt character and per item in the main study, if its
    results are on this computer (same model, instructions and schema)."""
    res, sets = Path(f"build/judge/results/{judge}/main.jsonl"), Path("build/judge/sets/main.jsonl")
    if not res.exists() or not sets.exists():
        return None
    reqs = {r["id"]: r for r in J._jsonl(sets)}
    chars = items = tin = tout = 0
    for r in J._jsonl(res):
        u = r["result"].get("usage") or {}
        if r["id"] in reqs and u.get("output"):
            chars += len(reqs[r["id"]]["prompt"]) + len(J.INSTRUCTIONS)
            items += len(reqs[r["id"]]["fids"])
            tin += (u.get("input") or 0) + (u.get("cache_read") or 0)
            tout += u["output"]
    return {"in_per_char": tin / chars, "out_per_item": tout / items} if items else None


def estimate(judges: list[str]) -> float:
    prices = json.loads(Path("config/prices.json").read_text(encoding="utf-8"))
    reqs = [r for name in SETS for r in J._jsonl(J.OUT / f"sets/{name}.jsonl")]
    chars = sum(len(r["prompt"]) + len(J.INSTRUCTIONS) for r in reqs)
    items = sum(len(r["fids"]) for r in reqs)
    total = 0.0
    for j in judges:
        model = J.JUDGES[j]
        rate = measured_rate(j)
        src = "measured on the main study" if rate else "rough default"
        rate = rate or DEFAULT_RATE[j]
        p = prices[model]
        cost = (chars * rate["in_per_char"] * p["input"] + items * rate["out_per_item"] * p["output"]) / 1e6
        cost *= prices["batch_discount"]
        total += cost
        print(f"  {j} ({model}): {len(reqs)} requests (main + planted errors + retest), {items} items — about ${cost:.2f} ({src})")
    return total


def main(argv: list[str]) -> None:
    judges = ["claude", "gpt"]
    if "--judges" in argv:
        judges = [j.strip() for j in argv[argv.index("--judges") + 1].split(",") if j.strip()]
        if not set(judges) <= set(J.JUDGES):
            raise SystemExit(f"--judges: choose from {', '.join(J.JUDGES)}")
    if "--skip-prepare" not in argv:
        prepare.main(["prepare"])
    J.OUT = prepare.BUILD / "judge"  # the judge module writes its sets, state and results here
    if "--mock" in argv:
        report.main(["report", "--mock"])
        return
    total = estimate(judges)
    print(f"Estimated total: about ${total:.2f} (Batch API prices from config/prices.json; "
          f"allow up to 1.5× for safety). The SPS text is sent to the judges' providers.")
    if input("Type yes to start: ").strip().lower() != "yes":
        raise SystemExit("not started")
    for j in judges:
        for name in SETS:
            J.submit_only(j, name)  # all batches run at the same time
    for j in judges:
        for name in SETS:
            J.run(j, name)
    report.main(["report"])


if __name__ == "__main__":
    main(sys.argv)
