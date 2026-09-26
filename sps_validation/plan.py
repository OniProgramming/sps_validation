"""Size the study to the budget: a random sample of Genesis sentences + all of Ephesians.

The cost per request is measured on the pilot (build/judge/pilot/<judge>/), priced
with config/prices.json at batch rates. The largest Genesis sample whose total
cost (main + retest + planted errors) stays within the budget for *both* judges
is drawn with a fixed seed, so the sample is reproducible and not chosen by hand.

    python -m sps_validation.plan estimate [--budget 12]   # print the plan, write nothing
    python -m sps_validation.plan write    [--budget 12]   # write sets main / retest / perturb
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from .judge import JUDGES, OUT, _jsonl, write_set
from .validate import KINDS, build_perturbations, build_retest

SEED = 20260925
PER_CELL = 10  # planted errors per translation × error type (4 × 6 × 10 = 240, + 240 controls)
STEP = 25


def cost_per_request() -> dict[str, float]:
    prices = json.loads(Path("config/prices.json").read_text(encoding="utf-8"))
    out = {}
    for judge, model in JUDGES.items():
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in (OUT / "pilot" / judge).glob("*.json")]
        runs = [r for r in runs if r.get("model") == model and r["result"].get("usage")]
        if not runs:
            raise SystemExit(f"no pilot results for {judge} ({model}); run the pilot first")
        if model not in prices:
            raise SystemExit(f"no price for {model} in config/prices.json")
        p = prices[model]
        cost = [((r["result"]["usage"].get("input") or 0) + (r["result"]["usage"].get("cache_read") or 0)) * p["input"]
                + (r["result"]["usage"].get("output") or 0) * p["output"] for r in runs]
        out[judge] = sum(cost) / len(cost) / 1e6 * prices["batch_discount"]
    return out


def genesis_order() -> list[str]:
    units = [json.loads(line)["unit_id"] for line in open("build/sources/GEN.units.jsonl", encoding="utf-8")]
    random.Random(SEED).shuffle(units)
    return units


def main_set(all_reqs: list[dict], sample: set[str]) -> list[dict]:
    return [r for r in all_reqs if r["book"] == "EPH" or any(u in sample for u in r["units"])]


def plan(budget: float) -> dict:
    all_reqs = _jsonl(OUT / "sets" / "all.jsonl")
    price = cost_per_request()
    order = genesis_order()
    extra = 2 * 4 * len(KINDS) * PER_CELL  # planted errors + their unperturbed controls
    best = None
    for n in range(STEP, len(order) + STEP, STEP):
        sample = set(order[:n])
        main = main_set(all_reqs, sample)
        total = len(main) + round(0.10 * len(main)) + extra
        cost = {j: total * c for j, c in price.items()}
        if all(v <= budget for v in cost.values()):
            best = {"n": min(n, len(order)), "requests_main": len(main), "requests_total": total,
                    "cost": {j: round(v, 2) for j, v in cost.items()}}
        else:
            break
    if not best:
        raise SystemExit(f"budget ${budget} is too small even for {STEP} Genesis sentences")
    best["per_request"] = {j: round(c, 5) for j, c in price.items()}
    best["models"] = dict(JUDGES)
    best["budget_per_judge"] = budget
    return best


def write(budget: float) -> dict:
    p = plan(budget)
    order = genesis_order()
    sample = order[: p["n"]]
    all_reqs = _jsonl(OUT / "sets" / "all.jsonl")
    main = main_set(all_reqs, set(sample))
    write_set("main", main)
    write_set("perturb", build_perturbations(PER_CELL))
    write_set("retest", build_retest())
    (OUT / "sample.json").write_text(json.dumps({"seed": SEED, "genesis_sentences": sample, "plan": p},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def describe(p: dict) -> str:
    return (f"Plan: {p['n']} randomly chosen Genesis sentences (of 4,220) + all 78 Ephesians sentences, "
            f"in all 4 translations.\n"
            f"Requests per judge: {p['requests_total']} (main {p['requests_main']}, 10% repeat, 240 planted errors + 240 controls).\n"
            + "\n".join(f"Estimated cost {j} ({p['models'][j]}): ${p['cost'][j]:.2f}  "
                        f"(${p['per_request'][j]:.4f} per request)" for j in p["cost"])
            + f"\nBudget per account: ${p['budget_per_judge']:.2f}")


def main(argv: list[str]) -> None:
    budget = float(argv[argv.index("--budget") + 1]) if "--budget" in argv else 12.0
    cmd = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "estimate"
    p = write(budget) if cmd == "write" else plan(budget)
    print(describe(p))


if __name__ == "__main__":
    main(sys.argv)
