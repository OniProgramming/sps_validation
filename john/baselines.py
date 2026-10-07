"""Baselines on John 1: COMETKiwi, xCOMET and GEMBA-MQM, as for Genesis and Ephesians.

The unchanged sps_validation.baselines (export, COMET runner, GEMBA-MQM prompt and scoring,
detection, neutral-edit calibration, McNemar, rankings) applied to the John 1 sets. Only the
folders change: inputs build/john/judge/, outputs build/john/baselines/ and
build/john/report/baselines/. Definitions: docs/BASELINES.md (fixed before any baseline was run).

    python -m john.baselines export              # 1. items.jsonl, pairs.jsonl, neutral.jsonl (free)
    python -m john.baselines gemba gpt --set pairs     # 2. GEMBA-MQM with GPT-5 mini (cost shown first)
    python -m john.baselines gemba gpt --set neutral
    (COMET models: Kaggle notebook, see john/PASI_BASELINES.md)
    python -m john.baselines report              # 3. tables → build/john/report/baselines/
    python -m john.baselines report --mock       #    pipeline test on mock scores
"""

from __future__ import annotations

import csv
import json
import sys

from sps_validation import baselines as B
from sps_validation import report as R
from sps_validation.judge import TRANSLATIONS, _jsonl
from sps_validation.validate import KINDS

from .prepare import BUILD, CODE, load_units
from .report import LABEL, write_mock

JUDGE_DIR = BUILD / "judge"
_rankings = B.rankings


def _john_rankings(items: dict, scores: dict, sate_rows: dict) -> list[dict]:
    """baselines.rankings iterates over the books GEN and EPH; John 1 is passed through it under
    the label EPH and relabelled afterwards (the computation is unchanged)."""
    items2 = {k: dict(i, book="EPH") for k, i in items.items()}
    rows2 = {d: [dict(r, book="EPH") for r in rows] for d, rows in sate_rows.items()}
    res = _rankings(items2, scores, rows2)
    for r in res:
        r["book"] = CODE
    return res


def use_john() -> None:
    """Point the baselines module at John 1 (inputs, outputs and labels only)."""
    B.OUT = JUDGE_DIR
    B.BASE = BUILD / "baselines"
    B.load_sources = load_units
    B.restrict_to_sample = lambda rows: rows  # John 1 is judged whole
    B.LANG = {CODE: "Koine Greek"}
    B.BOOK_LABEL = {CODE: LABEL}
    B.rankings = _john_rankings


def report(mock: bool) -> None:
    """As baselines.report, with John 1's folders."""
    pairs = B._read(B.BASE / "pairs.jsonl")
    items = {i["key"]: i for i in B._read(B.BASE / "items.jsonl")}
    if not pairs:
        raise SystemExit("run `python -m john.baselines export` first")
    if mock:
        root, sroot, out, label = JUDGE_DIR / "results_mock", B.BASE / "scores_mock", \
            BUILD / "report_mock" / "baselines", "John 1 — MOCK DATA, pipeline test, not a result"
        if not (root / "claude" / "main.jsonl").exists():
            write_mock(root)
        B.write_mock_scores(items, sroot)
    else:
        root, sroot, out, label = JUDGE_DIR / "results", B.BASE / "scores", BUILD / "report" / "baselines", LABEL
    sate = {"pert": R.load_results(root, "perturb"), "main": R.load_results(root, "main")}
    scores = B.load_scores(sroot)
    records = B.evaluate(pairs, items, scores, sate)
    table = B.rate_table(records)
    neutral = B._read(B.BASE / "neutral.jsonl")
    calib = B.calibration(records, B.neutral_deltas(neutral, scores))
    nspans = B.neutral_spans(neutral, items, scores)
    ref = "SATE Claude+GPT · targeted · sense"
    tests = B.mcnemar(records, ref) if ref in records else []
    units, feats = load_units()
    main_reqs = _jsonl(JUDGE_DIR / "sets" / "main.jsonl")
    sate_rows = {d: R.sentence_rows(main_reqs, sate["main"], units, feats, k)
                 for d, k in R.DIMENSIONS.items()} if sate["main"] else {}
    ranks = B.rankings(items, scores, sate_rows)
    res = {"label": label, "methods": list(records), "detection": table, "mcnemar_vs": ref, "mcnemar": tests,
           "calibrated": calib, "neutral_spans": nspans, "neutral_controls": len(neutral),
           "rankings": ranks, "pairs": len(pairs)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=B._plain),
                                        encoding="utf-8")
    B._write_jsonl(out / "pair_records.jsonl", ({"method": m} | r for m, rs in records.items() for r in rs))
    with (out / "baselines_detection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [f"{k} detection" for k in KINDS] + ["All: detection", "All: false alarms",
                                                                     "All: mean Δ", "All: p(Δ>0)", "Pairs", "Truncated"])
        for r in table:
            a = r["all"]
            w.writerow([r["method"]] + [r[k]["rate"] for k in KINDS] +
                       [a["rate"], a["false_alarm_rate"], a["mean_delta"], a["p_delta_gt_0"], a["n"], r["truncated"]])
    with (out / "baselines_calibrated.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [f"{k} detection" for k in KINDS] + ["All: detection", "Threshold τ",
                                                                     "False alarms at τ", "Null comparisons",
                                                                     "Neutral edits with any drop"])
        for r in calib:
            w.writerow([r["method"]] + [r[k] for k in KINDS] + [r["all"], r["tau"], r["null_false_alarm"], r["null_n"],
                                                                r["neutral_any_drop"]])
    with (out / "baselines_rankings.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Book"] + list(TRANSLATIONS) + ["Order", "Kendall W", "Significant pairs",
                                                              "τ with SATE sense", "τ with SATE source"])
        for r in ranks:
            w.writerow([r["method"], LABEL] + [r["means"].get(t, "") for t in TRANSLATIONS] +
                       [r["order"], r["W"], r["significant"], r.get("tau_sense", ""), r.get("tau_source", "")])
    md = B.markdown(res)
    # the generic text quotes the main study's SATE false-alarm rate; John 1 has its own (table above)
    md = md.replace("SATE's targeted verdicts need no threshold (false alarms 1.5%).",
                    "SATE's targeted verdicts need no threshold (their false alarms are in the first table).")
    (out / "baselines.md").write_text(md, encoding="utf-8")
    print(f"{len(table)} methods compared on {len(pairs)} planted-error pairs; results in {out.resolve()}")


def main(argv: list[str]) -> None:
    use_john()
    cmd = argv[1] if len(argv) > 1 else "help"
    if cmd == "report":
        report("--mock" in argv)
    elif cmd in ("export", "comet", "gemba"):
        B.main(argv)  # unchanged commands, now on John 1's folders
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
