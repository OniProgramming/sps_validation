"""Report for John 1: the scoring and statistics of the main study, applied to John 1.

Per-sentence scores (report.sentence_rows), totals with bootstrap CIs (report.totals),
Friedman / Kendall's W / Wilcoxon-Holm (report.comparisons), judge agreement
(report.agreement), rule cross-checks and the four article tables (report.article_tables)
are the unchanged functions of sps_validation.report. Output: build/john/report/.

    python -m john.report            # real results in build/john/judge/results/
    python -m john.report --mock     # pipeline test on random judgements (clearly labelled)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from sps_validation import report as R
from sps_validation.judge import TRANSLATIONS, _jsonl

from .prepare import BUILD, CODE, load_units

JUDGE_DIR = BUILD / "judge"
LABEL = "John 1"


def write_mock(root: Path) -> None:
    """Random judgements, for testing the pipeline only (as report.write_mock)."""
    rng = random.Random(1)
    outcomes = ["retained"] * 7 + ["partial", "lost", "distorted"]
    reqs = _jsonl(JUDGE_DIR / "sets/main.jsonl")
    for j in R.JUDGES:
        (root / j).mkdir(parents=True, exist_ok=True)
        with (root / j / "main.jsonl").open("w", encoding="utf-8") as f:
            for r in reqs:
                res = {"status": "ok", "missing": [],
                       "features": [{"fid": x, "outcome": rng.choice(outcomes), "source_outcome": rng.choice(outcomes),
                                     "english": "", "transliterated": r["translation"] == "SPS" and rng.random() < 0.1,
                                     "displaced": False, "reason": "MOCK"} for x in r["fids"]],
                       "additions": [{"english": "", "type": rng.choice(["grammatical", "unsupported"]),
                                      "reason": "MOCK"} for _ in range(rng.randint(0, 2))]}
                meta = {k: r[k] for k in ("id", "translation", "book", "units")}
                f.write(json.dumps(meta | {"model": "MOCK", "result": res}) + "\n")


def _relabel(tables: dict) -> dict:
    """article_tables names the pooled rows "Both books"; here they are John 1."""
    for t in tables.values():
        t["title"] = t["title"].replace("both books", LABEL)
        t["rows"] = [[LABEL if c == "Both books" else c for c in row] for row in t["rows"]]
        t["notes"] = [n.replace("Both books", LABEL) for n in t.get("notes", [])]
    return tables


def build(root: Path, out: Path, label: str) -> dict:
    units, feats = load_units()
    feats_by_fid = {f["fid"]: f for fl in feats.values() for f in fl}
    reqs = _jsonl(JUDGE_DIR / "sets/main.jsonl")
    main = R.load_results(root, "main")
    if not main:
        raise SystemExit(f"no judge results in {root}")
    out.mkdir(parents=True, exist_ok=True)

    rows_by = {d: R.sentence_rows(reqs, main, units, feats, k) for d, k in R.DIMENSIONS.items()}
    R._write_csv(out / "sentences.csv", R.merge_dimensions(rows_by))
    served = {j: sorted({(x["result"].get("usage") or {}).get("served_model") or x["model"] for x in r.values()})
              for j, r in main.items()}
    summary = {"label": label, "judges": {j: ", ".join(v) for j, v in served.items()},
               "status": {j: dict(Counter(x["result"]["status"] for x in r.values())) for j, r in main.items()},
               "dimensions": {d: {"totals": R.totals(r), "comparisons": R.comparisons(r)} for d, r in rows_by.items()}}
    summary["per_judge"] = {
        j: {d: {"totals": R.totals(rj), "comparisons": R.comparisons(rj)}
            for d, k in R.DIMENSIONS.items()
            for rj in [R.sentence_rows(reqs, {j: main[j]}, units, feats, k)]}
        for j in main}
    if len(main) == 2:
        summary["judge_agreement"] = {d: R.agreement(main["claude"], main["gpt"], reqs, feats_by_fid, k)
                                      for d, k in R.DIMENSIONS.items()}
    summary["rule_crosscheck"] = R.rule_crosscheck(reqs, main, units, feats)
    summary["unaligned_english"] = {
        t: {"pieces": sum(len(r.get("unaligned_english", [])) for r in reqs if r["translation"] == t),
            "words": sum(len(" ".join(r.get("unaligned_english", [])).split()) for r in reqs if r["translation"] == t)}
        for t in TRANSLATIONS}
    align = BUILD / "align/summary.json"
    if align.exists():
        summary["alignment"] = json.loads(align.read_text(encoding="utf-8"))
    tables = _relabel(R.article_tables(summary, rows_by))
    summary["article_tables"] = tables
    for name, t in tables.items():
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(t["columns"])
            w.writerows(t["rows"])
    (out / "report.md").write_text(R.tables_markdown(tables) + "\n" + markdown(summary), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def markdown(s: dict) -> str:
    L = [f"# Details — {s['label']}", "",
         "Judges: " + ", ".join(f"{j} = `{m}`" for j, m in s["judges"].items()),
         "Judge answers: " + "; ".join(f"{j} {dict(v)}" for j, v in s["status"].items()), ""]
    head = ("| Translation | Sentences | Items | Retention [95% CI] | Accuracy | Fidelity [95% CI] | "
            "Distorted | Unsupported additions | Transliterated items | Their score |")
    for d in ("sense", "source"):
        for who, block in [("both judges", s["dimensions"][d])] + [(j, s["per_judge"][j][d]) for j in s["per_judge"]]:
            if who != "both judges" and len(s["per_judge"]) == 1:
                continue
            T, C = block["totals"], block["comparisons"].get("ALL")
            L += [f"## {R.DIM_LABEL[d]} — {who}", "", head, "|" + "---|" * 10]
            for t in TRANSLATIONS:
                x = T.get(f"ALL.{t}")
                if x:
                    ts = "—" if x["transliterated_score"] is None else f"{x['transliterated_score']:.3f}"
                    L.append(f"| {t} | {x['sentences']} | {x['features']:.0f} | {x['retention']:.3f} "
                             f"[{x['retention_CI95'][0]:.3f}, {x['retention_CI95'][1]:.3f}] | {x['accuracy']:.3f} | "
                             f"{x['fidelity']:.3f} [{x['fidelity_CI95'][0]:.3f}, {x['fidelity_CI95'][1]:.3f}] | "
                             f"{x['distorted']} | {x['unsupported_additions']} | {x['transliterated_features']} | {ts} |")
            if C:
                L += ["", f"Friedman χ² = {C['friedman_chi2']}, p = {C['friedman_p']:.3g}, "
                          f"Kendall's W = {C['kendall_W']}, n = {C['sentences']}. Pairs (Holm): " +
                      "; ".join(f"{p['pair']} Δ̃ = {p['median_diff']}, r = {p['r_rb']}, p = {p['p_holm']:.3g}"
                                for p in C["pairwise"])]
            L.append("")
    if "judge_agreement" in s:
        L += ["## Agreement between the judges (Krippendorff's α, all items)", ""]
        for d, a in s["judge_agreement"].items():
            x = a.get("ALL", {})
            alpha = "—" if x.get("alpha") is None else f"{x['alpha']:.3f}"
            L.append(f"- {R.DIM_LABEL[d]}: α = {alpha}, exact agreement {x.get('exact_agreement')}, n = {x.get('n')}")
        L.append("")
    if s.get("alignment"):
        L += ["## Alignment (verse-free aligner; verse numbers used only to check it)", "",
              "| Translation | English pieces | Sentences in the right verse | Pieces in the right verse |", "|---|---|---|---|"]
        for t, a in s["alignment"].items():
            L.append(f"| {t} | {a['pieces']} | {a.get('in_right_verse')}/{a.get('units')} ({a.get('accuracy')}) | "
                     f"{a.get('piece_in_right_verse')} |")
        L.append("")
    L += ["## Unaligned English (judged as possible additions)", ""] + \
         [f"- {t}: {v['pieces']} pieces, {v['words']} words" for t, v in s["unaligned_english"].items()] + [""]
    if s.get("rule_crosscheck"):
        L += ["## Rule cross-checks (agreement of the judge with a simple rule)", ""] + \
             [f"- {k}: {v}" for k, v in sorted(s["rule_crosscheck"].items())] + [""]
    return "\n".join(L)


def main(argv: list[str]) -> None:
    if "--mock" in argv:
        root = JUDGE_DIR / "results_mock"
        write_mock(root)
        s = build(root, BUILD / "report_mock", "MOCK — random judgements, pipeline test only")
    else:
        s = build(JUDGE_DIR / "results", BUILD / "report", LABEL)
    print(R.tables_markdown({"table1": s["article_tables"]["table1"]}))
    print("written", (BUILD / ("report_mock" if "--mock" in argv else "report")).resolve())


if __name__ == "__main__":
    main(sys.argv)
