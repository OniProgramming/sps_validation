"""Judge the word-order items (ORD) with the main study's judges, then report.

Every request of the main run that contains an ORD item is sent again with only its ORD items:
the same source sentence, the same English, the same context and text notes. The judges are
those of the main study (models, settings, schema, Batch APIs); their instructions are the main
study's, followed by a section on ORD (ORD_SECTION below). A 10% retest is judged too.

    python -m order.run              # Genesis + Ephesians, in the main study's folder
    python -m order.run --john       # John 1, in the John folder (after john.run)
    python -m order.run --judges gpt # one judge only
    python -m order.run --mock       # pipeline test, no API calls

The run can be interrupted: the same command resumes the submitted batches.
Output: build/order/ (Genesis + Ephesians) or build/john/order/ (John 1).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from sps_validation import judge as J

from .items import unit_items

SETS = ("main", "retest")
SEED = 20261008

ORD_SECTION = """

## ORD features (word order)
In this request FEATURES contains only ORD features. An ORD feature states that a constituent of a source clause stands before the verb of that clause. Hebrew and Greek order the constituents of a clause freely; a constituent placed before the verb is there to give it prominence: it sets the frame or the topic of the clause, or it is in focus. The feature is this placement and the prominence it gives in context. The feature names the constituent, the verb and the order of the whole clause.
For ORD features the statement above that word order is irrelevant for SENSE does not apply, and rules 6-14 are replaced by these:
- outcome (SENSE CONVEYED): retained when an ordinary reader of the English perceives the constituent as prominent in the same way (as the frame or topic, or as the focus), by any means: placing it first, a cleft ("it was X that ..."), "as for X", an emphasising word. partial: the prominence is weaker or only implied. lost: the constituent stands in an ordinary position and nothing else marks it. distorted: the English order or construction makes the reader take the constituent in a different grammatical role (for example a predicate read as the subject) or attach it to a different word.
- source_outcome (SOURCE PRESERVED): retained when the English keeps the constituent before its verb, in the source's order relative to the other constituents of the clause. partial: only part of the constituent stands before the verb, or its order relative to the other constituents is kept only in part. lost: the constituent stands after its verb. distorted: the English order changes who does what, or which word the constituent belongs to.
- If the English has no counterpart of the constituent, both verdicts are lost.
- If the TEXT NOTES show that this translation's source edition lacks the constituent or orders these words differently, use not_in_base.
- "english": the English words of the constituent. Return an empty additions list: additions are judged in the main run."""


def dataset(john: bool) -> dict:
    """Where the units, the MACULA trees and the main run of this folder are."""
    if john:
        from john.prepare import BUILD, SOURCES_DIR, load_units

        units, feats = load_units()
        return {"label": "John 1", "units": units, "feats": feats, "keep": None,
                "trees": [SOURCES_DIR / "macula-greek/SBLGNT/lowfat/04-john.xml"],
                "main": BUILD / "judge", "out": BUILD / "order"}
    units, feats = J.load_sources()
    sample = Path("build/judge/sample.json")
    keep = None
    if sample.exists():  # the Genesis sentences of the main study and all of Ephesians
        keep = set(json.loads(sample.read_text(encoding="utf-8"))["genesis_sentences"])
        keep |= {u for u in units if u.startswith("EPH")}
    src = Path("data/sources")
    trees = sorted((src / "macula-hebrew/WLC/lowfat").glob("01-Gen-*-lowfat.xml"))
    trees.append(src / "macula-greek/SBLGNT/lowfat/10-ephesians.xml")
    return {"label": "Genesis and Ephesians", "units": units, "feats": feats, "keep": keep,
            "trees": trees, "main": Path("build/judge"), "out": Path("build/order")}


def use_ord(ds: dict) -> None:
    """Point the judge module at this run: its folder and the instructions with ORD_SECTION."""
    J.OUT = ds["out"] / "judge"
    if not J.INSTRUCTIONS.endswith(ORD_SECTION):
        J.INSTRUCTIONS += ORD_SECTION


def load_items(ds: dict) -> dict[str, list[dict]]:
    return {r["unit_id"]: r["features"] for r in J._jsonl(ds["out"] / "items.jsonl")}


def prepare(ds: dict) -> None:
    missing = [p for p in ds["trees"] if not p.exists()]
    if missing:
        raise SystemExit(f"MACULA files not found: {missing[0]} … (run this in the folder of the study)")
    main_set = ds["main"] / "sets/main.jsonl"
    if not main_set.exists():
        raise SystemExit(f"{main_set} not found: run the main study in this folder first")
    items = unit_items(ds["units"], ds["trees"], ds["keep"])
    ds["out"].mkdir(parents=True, exist_ok=True)
    with (ds["out"] / "items.jsonl").open("w", encoding="utf-8") as f:
        for uid in ds["units"]:
            if uid in items:
                f.write(json.dumps({"unit_id": uid, "features": items[uid]}, ensure_ascii=False) + "\n")
    reqs = []
    for r in J._jsonl(main_set):
        fl = [f for u in r["units"] for f in items.get(u, [])]
        if not fl:
            continue
        req = {k: r[k] for k in ("translation", "book", "units", "english", "before", "after", "edition")}
        req |= {"id": "o" + hashlib.sha256(f"ORD|{r['id']}".encode()).hexdigest()[:20], "main_id": r["id"],
                "fids": [f["fid"] for f in fl]}
        req["prompt"] = J.render([ds["units"][u] for u in r["units"]], fl, r["english"], r["before"],
                                 r["after"], r["edition"])
        reqs.append(req)
    rng = random.Random(SEED)
    retest = []
    for t in J.TRANSLATIONS:
        rs = [r for r in reqs if r["translation"] == t]
        retest += rng.sample(rs, max(1, round(len(rs) * 0.10))) if rs else []
    J.write_set("main", reqs)
    J.write_set("retest", retest)
    n = sum(len(v) for v in items.values())
    print(f"{ds['label']}: {n} ORD items in {len(items)} sentences; "
          f"{len(reqs)} requests per judge (+ {len(retest)} retest)")


def measured_rate(ds: dict, judge: str) -> dict | None:
    """Tokens per prompt character and per item actually used by this judge in the main run."""
    res, sets = ds["main"] / f"results/{judge}/main.jsonl", ds["main"] / "sets/main.jsonl"
    if not res.exists() or not sets.exists():
        return None
    reqs = {r["id"]: r for r in J._jsonl(sets)}
    base = len(J.INSTRUCTIONS) - (len(ORD_SECTION) if J.INSTRUCTIONS.endswith(ORD_SECTION) else 0)
    chars = items = tin = tout = 0
    for r in J._jsonl(res):
        u = r["result"].get("usage") or {}
        if r["id"] in reqs and u.get("output"):
            chars += len(reqs[r["id"]]["prompt"]) + base
            items += len(reqs[r["id"]]["fids"])
            tin += (u.get("input") or 0) + (u.get("cache_read") or 0)
            tout += u["output"]
    return {"in_per_char": tin / chars, "out_per_item": tout / items} if items else None


DEFAULT_RATE = {"claude": {"in_per_char": 1 / 3.2, "out_per_item": 90},
                "gpt": {"in_per_char": 1 / 3.2, "out_per_item": 260}}


def estimate(ds: dict, judges: list[str]) -> float:
    prices = json.loads(Path("config/prices.json").read_text(encoding="utf-8"))
    reqs = [r for name in SETS for r in J._jsonl(J.OUT / f"sets/{name}.jsonl")]
    chars = sum(len(r["prompt"]) + len(J.INSTRUCTIONS) for r in reqs)
    items = sum(len(r["fids"]) for r in reqs)
    total = 0.0
    for j in judges:
        model = J.JUDGES[j]
        rate = measured_rate(ds, j)
        src = "measured on the main run" if rate else "rough default"
        rate = rate or DEFAULT_RATE[j]
        p = prices[model]
        cost = (chars * rate["in_per_char"] * p["input"] + items * rate["out_per_item"] * p["output"]) / 1e6
        cost *= prices["batch_discount"]
        total += cost
        print(f"  {j} ({model}): {len(reqs)} requests (main + retest), {items} items — about ${cost:.2f} ({src})")
    return total


def main(argv: list[str]) -> None:
    from . import report

    judges = ["claude", "gpt"]
    if "--judges" in argv:
        judges = [j.strip() for j in argv[argv.index("--judges") + 1].split(",") if j.strip()]
        if not set(judges) <= set(J.JUDGES):
            raise SystemExit(f"--judges: choose from {', '.join(J.JUDGES)}")
    ds = dataset("--john" in argv)
    use_ord(ds)
    prepare(ds)
    if "--mock" in argv:
        report.write_mock(ds)
        report.main(argv + ["--mock"])
        return
    total = estimate(ds, judges)
    print(f"Estimated total: about ${total:.2f} (Batch API prices from config/prices.json; "
          f"allow up to 1.5× for safety). The SPS text is sent to the judges' providers.")
    if input("Type yes to start: ").strip().lower() != "yes":
        raise SystemExit("not started")
    for j in judges:
        for name in SETS:
            J.submit_only(j, name)
    for j in judges:
        for name in SETS:
            J.run(j, name)
    report.main(argv)


if __name__ == "__main__":
    main(sys.argv)
