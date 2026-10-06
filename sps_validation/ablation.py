"""Ablation studies on the existing judge results — no new API calls, no cost.

Each ablation changes one component of the scoring and recomputes the same
quantities from the same categorical verdicts:

  A1  outcome taxonomy: full (retained/partial/lost/distorted) vs binary
      preserved/not preserved (partial counted as preserved, or as not preserved)
  A2  dimension: sense only vs source only vs both (mean of the two fidelities)
  A3  feature family: leave one family out (lexical, verbal, referential, relational)
  A4  judges: Claude only, GPT only, combined
  A5  partial-retention weight w = 0.25 / 0.50 / 0.75
  A6  accuracy components: without unsupported additions; LOST and DISTORTED collapsed

For every variant: fidelity per book × dimension × translation, the order of the
translations, Kendall's W (Friedman), the number of significant pairwise
differences (Wilcoxon, Holm) and how many pairwise conclusions differ from the
primary analysis. Where the ablation also changes what counts as a detected
planted error (A1, A4), detection and false-alarm rates are recomputed.

    python -m sps_validation.ablation           # real results in build/judge/results/
    python -m sps_validation.ablation --mock    # pipeline test on mock judgements

Writes build/report/ablation/{ablation.md, ablation.json, ablation_rankings.csv,
ablation_detection.csv}.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from .judge import OUT, TRANSLATIONS, _jsonl, load_sources
from .report import (DIMENSIONS, DIM_LABEL, GROUPS, JUDGES, REPORT, comparisons, feature_outcomes, harmonic,
                     load_results, outcome_score, ratio, restrict_to_sample, sentence_rows, write_mock)
from .validate import KINDS

BOOKS = ("GEN", "EPH")
BOOK_LABEL = {"GEN": "Genesis", "EPH": "Ephesians"}
FAMILIES = {"Lexical": GROUPS["Lexical meaning (semantic field)"],
            "Verbal": GROUPS["Verbal grammar (aspect, stem, voice, mood)"],
            "Referential": GROUPS["Reference and number (person, number, definiteness)"],
            "Relational": GROUPS["Relations and syntax (relations, who-does-what, negation)"]}
PRIMARY = {"w": 0.5, "binary": None, "distortion": True, "additions": True, "exclude": frozenset()}


# --------------------------------------------------------------------------- scoring under a variant

def book_totals(rows: list[dict]) -> dict:
    """Item-weighted fidelity per book and translation (no bootstrap: point estimates only)."""
    out = {}
    for b in BOOKS:
        for t in TRANSLATIONS:
            rs = [r for r in rows if r["translation"] == t and r["book"] == b]
            if not rs:
                continue
            S = sum(r["sum_score"] for r in rs)
            N = sum(r["features"] for r in rs)
            D = sum(r["distorted"] for r in rs)
            U = sum(r["unsupported_additions"] for r in rs)
            R = S / N
            out[f"{b}.{t}"] = round(harmonic(ratio(S, S + D + U), R), 4)
    return out


def summarize(rows: list[dict]) -> dict:
    """Fidelity, order, W and pairwise conclusions for each book."""
    tot = book_totals(rows)
    comp = comparisons(rows)
    out = {}
    for b in BOOKS:
        f = {t: tot[f"{b}.{t}"] for t in TRANSLATIONS if f"{b}.{t}" in tot}
        c = comp.get(b)
        conclusions = {}
        if c:
            for p in c["pairwise"]:
                # the direction of a significant difference; 0 = no significant difference
                sign = (1 if p["r_rb"] > 0 else -1) if p["p_holm"] < 0.05 else 0
                conclusions[p["pair"]] = sign
        out[b] = {"fidelity": f, "order": " > ".join(sorted(f, key=lambda t: -f[t])),
                  "W": c["kendall_W"] if c else None, "conclusions": conclusions,
                  "significant": sum(v != 0 for v in conclusions.values())}
    return out


def combine_dimensions(sense: list[dict], source: list[dict]) -> list[dict]:
    """Rows whose fidelity is the mean of the two dimensions (A2 'both' only)."""
    src = {(r["sentence"], r["translation"]): r for r in source}
    out = []
    for r in sense:
        o = src.get((r["sentence"], r["translation"]))
        if o:
            out.append(dict(r, fidelity=round((r["fidelity"] + o["fidelity"]) / 2, 4)))
    return out


def summarize_both(sense: list[dict], source: list[dict]) -> dict:
    """A2 'both dimensions': per-sentence tests on the mean fidelity; book totals = mean of
    the two item-weighted fidelities."""
    s = summarize(combine_dimensions(sense, source))
    ts, to = book_totals(sense), book_totals(source)
    for b in BOOKS:
        f = {t: round((ts[f"{b}.{t}"] + to[f"{b}.{t}"]) / 2, 4) for t in TRANSLATIONS
             if f"{b}.{t}" in ts and f"{b}.{t}" in to}
        s[b]["fidelity"] = f
        s[b]["order"] = " > ".join(sorted(f, key=lambda t: -f[t]))
    return s


def changed(a: dict, ref: dict) -> tuple[int, int]:
    """Pairwise conclusions (significant + direction, or not significant) that differ from ref."""
    n = diff = 0
    for b in BOOKS:
        for pair, v in ref[b]["conclusions"].items():
            if pair in a[b]["conclusions"]:
                n += 1
                diff += a[b]["conclusions"][pair] != v
    return diff, n


# --------------------------------------------------------------------------- planted errors under a variant

def detection(pert_reqs: list[dict], pert: dict, main: dict, judges: tuple, key: str,
              binary: str | None = None) -> dict:
    """Detection of planted errors when the verdicts of `judges` are combined by their mean score
    (as in the main scoring). A case counts only if every judge returned it and the target was
    retained in the main run. Detected = combined score of the target below 1 in the perturbed
    text; false alarm = the same on the identical, unperturbed control judged in the same run.
    For additions: the mean number of unsupported additions rises above the main run."""
    cases = [q for q in pert_reqs if not q["perturbation"].get("control")]
    unsupported = lambda r: sum(a["type"] == "unsupported" for a in r["result"].get("additions", []))
    cell = defaultdict(lambda: [0, 0, 0])  # hits, alarms, cases
    for q in cases:
        ctrl = "c" + q["id"][1:]
        recs = []
        for j in judges:
            r = (pert.get(j, {}).get(q["id"]), pert.get(j, {}).get(ctrl), main.get(j, {}).get(q["base_id"]))
            if not all(r) or any(x["result"].get("status") != "ok" for x in r):
                break
            recs.append(r)
        else:
            kind, target = q["perturbation"]["kind"], q["perturbation"]["target"]
            if kind == "addition":
                mean = lambda i: sum(unsupported(r[i]) for r in recs) / len(recs)
                hit, alarm = mean(0) > mean(2), mean(1) > mean(2)
            else:
                outs = [tuple(feature_outcomes(x).get(target, {}).get(key) for x in r) for r in recs]
                if any(o is None or o not in ("retained", "partial", "lost", "distorted") for oo in outs for o in oo):
                    continue
                if any(o[2] != "retained" for o in outs):
                    continue
                score = lambda i: sum(outcome_score(o[i], 0.5, binary) for o in outs) / len(outs)
                hit, alarm = score(0) < 1, score(1) < 1
            c = cell[kind]
            c[0] += hit
            c[1] += alarm
            c[2] += 1
    rate = lambda a, n: round(a / n, 3) if n else None
    out = {k: {"cases": cell[k][2], "rate": rate(cell[k][0], cell[k][2]), "false_alarm_rate": rate(cell[k][1], cell[k][2])}
           for k in KINDS}
    tot = [sum(cell[k][i] for k in KINDS) for i in range(3)]
    out["all"] = {"cases": tot[2], "rate": rate(tot[0], tot[2]), "false_alarm_rate": rate(tot[1], tot[2])}
    return out


# --------------------------------------------------------------------------- the ablations

def run(root: Path, out: Path, label: str) -> dict:
    units, feats = load_sources()
    main_reqs = _jsonl(OUT / "sets" / "main.jsonl")
    main = load_results(root, "main")
    if not main:
        raise SystemExit(f"no judge results in {root}")
    pert = load_results(root, "perturb")
    pert_reqs = _jsonl(OUT / "sets" / "perturb.jsonl") if (OUT / "sets" / "perturb.jsonl").exists() else []
    judges_all = tuple(j for j in JUDGES if j in main)

    def rows(key: str, results: dict | None = None, **opt) -> list[dict]:
        cfg = PRIMARY | opt
        return restrict_to_sample(sentence_rows(main_reqs, results or main, units, feats, key, **cfg))

    primary = {d: rows(k) for d, k in DIMENSIONS.items()}
    ref = {d: summarize(r) for d, r in primary.items()}
    variants = []  # (ablation, variant, dimension, summary)

    def add(ablation, variant, d, summary):
        variants.append({"ablation": ablation, "variant": variant, "dimension": d, "summary": summary})

    for d in DIMENSIONS:
        add("A0 primary", "Full SATE (primary)", d, ref[d])
    # A1 taxonomy
    for d, k in DIMENSIONS.items():
        add("A1 taxonomy", "Binary, partial = preserved", d, summarize(rows(k, binary="lenient")))
        add("A1 taxonomy", "Binary, partial = not preserved", d, summarize(rows(k, binary="strict")))
    # A2 dimensions
    add("A2 dimension", "Both dimensions (mean)", "both", summarize_both(primary["sense"], primary["source"]))
    # A3 feature families
    for d, k in DIMENSIONS.items():
        for fam, classes in FAMILIES.items():
            add("A3 family", f"Without {fam.lower()}", d, summarize(rows(k, exclude=frozenset(classes))))
    # A4 judges
    for d, k in DIMENSIONS.items():
        for j in judges_all:
            add("A4 judges", f"{j.capitalize()} only", d, summarize(rows(k, {j: main[j]})))
    # A5 partial weight
    for d, k in DIMENSIONS.items():
        for w in (0.25, 0.75):
            add("A5 weight", f"w = {w}", d, summarize(rows(k, w=w)))
    # A6 accuracy components
    for d, k in DIMENSIONS.items():
        add("A6 accuracy", "Without unsupported additions", d, summarize(rows(k, additions=False)))
        add("A6 accuracy", "LOST and DISTORTED collapsed", d, summarize(rows(k, distortion=False)))

    for v in variants:
        r = ref.get(v["dimension"])
        v["changed"], v["pairs"] = changed(v["summary"], r) if r else (None, None)

    # family shares (how much of the score each family carries)
    shares = {}
    for d in DIMENSIONS:
        n = defaultdict(int)
        for r in primary[d]:
            for fam, classes in FAMILIES.items():
                n[fam] += sum(r[f"n_{c}"] for c in classes)
        total = sum(n.values())
        shares[d] = {fam: round(n[fam] / total, 3) for fam in FAMILIES} if total else {}

    det = []
    if pert:
        for d, k in DIMENSIONS.items():
            for jj in [(j,) for j in judges_all if j in pert] + ([judges_all] if len(judges_all) > 1 else []):
                name = " + ".join(x.capitalize() for x in jj) if len(jj) > 1 else f"{jj[0].capitalize()} only"
                for tax, binary in (("Full taxonomy", None), ("Binary, partial = preserved", "lenient")):
                    det.append({"dimension": d, "judges": name, "taxonomy": tax,
                                "detection": detection(pert_reqs, pert, main, jj, k, binary)})

    result = {"label": label, "family_shares": shares, "variants": variants, "detection": det}
    write(result, out)
    return result


# --------------------------------------------------------------------------- output

def write(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out / "ablation_rankings.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Ablation", "Variant", "Dimension", "Book"] + [f"{t} fidelity" for t in TRANSLATIONS] +
                   ["Order", "Kendall W", "Significant pairs", "Conclusions changed vs primary"])
        for v in res["variants"]:
            for b in BOOKS:
                s = v["summary"][b]
                w.writerow([v["ablation"], v["variant"], v["dimension"], BOOK_LABEL[b]] +
                           [round(100 * s["fidelity"][t], 1) if t in s["fidelity"] else "" for t in TRANSLATIONS] +
                           [s["order"], s["W"], s["significant"],
                            "" if v["changed"] is None else f"{v['changed']}/{v['pairs']}"])
    with (out / "ablation_detection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Dimension", "Judges", "Taxonomy"] + [f"{k} detection" for k in KINDS] +
                   ["All: detection", "All: false alarms", "Cases"])
        for x in res["detection"]:
            dd = x["detection"]
            w.writerow([x["dimension"], x["judges"], x["taxonomy"]] + [dd[k]["rate"] for k in KINDS] +
                       [dd["all"]["rate"], dd["all"]["false_alarm_rate"], dd["all"]["cases"]])
    (out / "ablation.md").write_text(markdown(res), encoding="utf-8")


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def markdown(res: dict) -> str:
    L = [f"# Ablation studies{' — ' + res['label'] if res['label'] else ''}", "",
         "All variants are recomputed from the same categorical verdicts as the primary analysis; "
         "no judge was re-run. 'Changed' counts pairwise conclusions (significant difference and its "
         "direction, or no significant difference; Wilcoxon, Holm, per book) that differ from the "
         "primary analysis of the same dimension.", ""]
    dims = {"sense": DIM_LABEL["sense"], "source": DIM_LABEL["source"], "both": "Both (mean)"}
    by_abl = defaultdict(list)
    for v in res["variants"]:
        by_abl[v["ablation"]].append(v)
    for abl, vs in by_abl.items():
        L += [f"## {abl}", "", "| Variant | Dimension | Book | " + " | ".join(TRANSLATIONS) +
              " | Order | W | Sig. pairs | Changed |", "|---|---|---|" + "---|" * len(TRANSLATIONS) + "---|---|---|---|"]
        for v in vs:
            for b in BOOKS:
                s = v["summary"][b]
                L.append(f"| {v['variant']} | {dims[v['dimension']]} | {BOOK_LABEL[b]} | " +
                         " | ".join(pct(s["fidelity"].get(t)) for t in TRANSLATIONS) +
                         f" | {s['order']} | {s['W']} | {s['significant']}/6 | "
                         f"{'—' if v['changed'] is None else str(v['changed']) + '/' + str(v['pairs'])} |")
        L.append("")
    if res["family_shares"]:
        L += ["Share of information items per family (primary analysis):", ""]
        for d, sh in res["family_shares"].items():
            L.append(f"- {DIM_LABEL[d]}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sh.items()))
        L.append("")
    if res["detection"]:
        L += ["## Planted errors under the ablations (A1 taxonomy, A4 judges)", "",
              "Combined judges: the mean of the judges' scores, as in the main scoring (a target counts as "
              "detected when at least one judge marks it not fully retained). Cases: target retained in the "
              "main run by every judge included.", "",
              "| Dimension | Judges | Taxonomy | " + " | ".join(KINDS) + " | All | False alarms | Cases |",
              "|---|---|---|" + "---|" * len(KINDS) + "---|---|---|"]
        for x in res["detection"]:
            dd = x["detection"]
            L.append(f"| {DIM_LABEL[x['dimension']]} | {x['judges']} | {x['taxonomy']} | " +
                     " | ".join("—" if dd[k]["rate"] is None else f"{dd[k]['rate']:.2f}" for k in KINDS) +
                     f" | {dd['all']['rate']} | {dd['all']['false_alarm_rate']} | {dd['all']['cases']} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> None:
    if "--mock" in argv:
        root = OUT / "results_mock"
        if not (root / "claude" / "main.jsonl").exists():
            write_mock(root)
        res = run(root, REPORT / "mock" / "ablation", "MOCK DATA — pipeline test, not a result")
    else:
        res = run(OUT / "results", REPORT / "ablation", "")
    print(f"{len(res['variants'])} variants; results in build/report/{'mock/' if '--mock' in argv else ''}ablation/")


if __name__ == "__main__":
    main(sys.argv)
