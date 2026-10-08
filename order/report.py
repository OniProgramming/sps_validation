"""Report of the word-order run: ORD alone, and the main scores with ORD added.

Scoring, statistics and agreement are those of sps_validation.report, unchanged: retained 1,
partial 0.5, lost 0, distorted 0, averaged over the judges; R, P, F; bootstrap intervals;
Friedman and Wilcoxon–Holm. Additions are not counted here (they belong to the main run).

    python -m order.report [--john] [--mock]

Output: build/order/report/ or build/john/order/report/. items.csv contains the SPS text:
keep it private.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter

from sps_validation import judge as J
from sps_validation import report as R

from .run import dataset, load_items, use_ord

TRANSLATIONS = J.TRANSLATIONS
ROLE_LABEL = {"o": "object", "o2": "second object", "io": "indirect object", "p": "predicate",
              "adv": "adverbial", "pp": "prepositional phrase"}


def write_mock(ds: dict) -> None:
    rng = random.Random(1)
    outcomes = ["retained"] * 5 + ["partial", "lost", "lost", "distorted"]
    root = J.OUT / "results_mock"
    for name in ("main", "retest"):
        for j in R.JUDGES:
            (root / j).mkdir(parents=True, exist_ok=True)
            with (root / j / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in J._jsonl(J.OUT / f"sets/{name}.jsonl"):
                    res = {"status": "ok", "missing": [], "additions": [],
                           "features": [{"fid": x, "outcome": rng.choice(outcomes),
                                         "source_outcome": rng.choice(outcomes), "english": "",
                                         "transliterated": False, "displaced": False, "reason": "MOCK"}
                                        for x in r["fids"]]}
                    meta = {k: r[k] for k in ("id", "translation", "book", "units")}
                    f.write(json.dumps(meta | {"model": "MOCK", "result": res}) + "\n")


def _all_units(ds: dict, items: dict) -> dict:
    return {u: items.get(u, []) for u in ds["units"]}


def rows(ds, reqs, res, items, key, judge=None):
    results = {judge: res[judge]} if judge else res
    return R.sentence_rows(reqs, results, ds["units"], _all_units(ds, items), key, additions=False)


def combined(main_rows: list[dict], ord_rows: list[dict]) -> list[dict]:
    """Main-run sentence rows with the ORD items of the same sentence and translation added."""
    extra = {(r["sentence"], r["translation"]): r for r in ord_rows}
    out = []
    for m in main_rows:
        o = extra.get((m["sentence"], m["translation"]))
        if not o:
            out.append(m)
            continue
        S, N = m["sum_score"] + o["sum_score"], m["features"] + o["features"]
        D, U = m["distorted"] + o["distorted"], m["unsupported_additions"]
        Rv, P = S / N, R.ratio(S, S + D + U)
        out.append(m | {"sum_score": S, "features": N, "distorted": D, "retention": Rv,
                        "accuracy": P, "fidelity": R.harmonic(P, Rv)})
    return out


def crosscheck(ds, reqs, res, items) -> dict:
    """Approximate check of the SOURCE verdicts: in the English, does the constituent (the judge's
    own words for it) come before the verb (the words the judges gave for the verb's LEX item in
    the main run)? Only cases where both are found once in the English are counted."""
    main_res = R.load_results(ds["main"] / "results", "main")
    if not main_res:
        return {}
    lex = {(f["token"]): f["fid"] for fl in ds["feats"].values() for f in fl if f["class"] == "LEX"}
    by_fid = {f["fid"]: f for fl in items.values() for f in fl}
    out = {}
    for j in res:
        if j not in main_res:
            continue
        agree = n = 0
        for r in reqs:
            if r["id"] not in res[j] or r["main_id"] not in main_res[j]:
                continue
            got, main = R.feature_outcomes(res[j][r["id"]]), R.feature_outcomes(main_res[j][r["main_id"]])
            text = r["english"].lower()
            for fid in r["fids"]:
                o, f = got.get(fid), by_fid[fid]
                v = main.get(lex.get(f["verb"][0], ""))
                if not o or not v or o["source_outcome"] not in ("retained", "partial", "lost"):
                    continue
                a, b = o["english"].lower().strip(), v["english"].lower().strip()
                if not a or not b or text.count(a) != 1 or text.count(b) != 1 or a in b or b in a:
                    continue
                before = text.index(a) < text.index(b)
                n += 1
                agree += before == (o["source_outcome"] != "lost")
        out[j] = {"cases": n, "agreement": round(agree / n, 4) if n else None}
    return out


def item_table(ds, reqs, res, items) -> list[list]:
    by_fid = {f["fid"]: (u, f) for u, fl in items.items() for f in fl}
    table = []
    for r in reqs:
        got = {j: R.feature_outcomes(res[j][r["id"]]) for j in res if r["id"] in res[j]}
        for fid in r["fids"]:
            u, f = by_fid[fid]
            row = [ds["units"][u]["refs"][0], u, r["translation"], ROLE_LABEL[f["role"]], f["value"], r["english"]]
            for j in R.JUDGES:
                o = got.get(j, {}).get(fid) or {}
                row += [o.get("outcome", ""), o.get("source_outcome", ""), o.get("english", ""), o.get("reason", "")]
            table.append(row)
    return table


def build(ds: dict, mock: bool) -> dict:
    items = load_items(ds)
    reqs = J._jsonl(J.OUT / "sets/main.jsonl")
    root = J.OUT / ("results_mock" if mock else "results")
    res = R.load_results(root, "main")
    if not res:
        raise SystemExit(f"no ORD results in {root}")
    out = ds["out"] / ("report_mock" if mock else "report")
    out.mkdir(parents=True, exist_ok=True)
    feats_by_fid = {f["fid"]: f for fl in items.values() for f in fl}
    s = {"label": ds["label"], "mock": mock,
         "items": {"total": sum(len(v) for v in items.values()), "sentences": len(items),
                   "by_role": dict(Counter(ROLE_LABEL[f["role"]] for fl in items.values() for f in fl))},
         "status": {j: dict(Counter(x["result"]["status"] for x in r.values())) for j, r in res.items()}}
    ord_rows = {d: rows(ds, reqs, res, items, k) for d, k in R.DIMENSIONS.items()}
    s["ord"] = {d: {"totals": R.totals(rs), "comparisons": R.comparisons(rs)} for d, rs in ord_rows.items()}
    s["ord_per_judge"] = {j: {d: R.totals(rows(ds, reqs, res, items, k, j)) for d, k in R.DIMENSIONS.items()}
                          for j in res}
    if len(res) == 2:
        s["judge_agreement"] = {d: R.agreement(res["claude"], res["gpt"], reqs, feats_by_fid, k).get("ALL")
                                for d, k in R.DIMENSIONS.items()}
    retest = R.load_results(root, "retest")
    rt = {r["id"] for r in J._jsonl(J.OUT / "sets/retest.jsonl")}
    s["test_retest"] = {d: {j: R.agreement(res[j], retest[j], [r for r in reqs if r["id"] in rt], feats_by_fid, k).get("ALL")
                            for j in retest if j in res} for d, k in R.DIMENSIONS.items()}
    s["source_crosscheck"] = crosscheck(ds, reqs, res, items)
    main_reqs = J._jsonl(ds["main"] / "sets/main.jsonl")
    main_res = R.load_results(ds["main"] / ("results_mock" if mock else "results"), "main")
    if main_res:
        s["with_main"] = {}
        for d, k in R.DIMENSIONS.items():
            m = R.sentence_rows(main_reqs, main_res, ds["units"], ds["feats"], k)
            if ds["keep"] is not None:
                m = [r for r in m if r["sentence"] in ds["keep"]]
            c = combined(m, ord_rows[d])
            s["with_main"][d] = {"main": R.totals(m), "main_plus_ord": R.totals(c),
                                 "comparisons": R.comparisons(c)}
    head = ["reference", "sentence", "translation", "role", "ORD item", "English"]
    for j in R.JUDGES:
        head += [f"{j} sense", f"{j} source", f"{j} English words", f"{j} reason"]
    with (out / "items.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(item_table(ds, reqs, res, items))
    (out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(markdown(s), encoding="utf-8")
    print("written", (out / "report.md").resolve())
    return s


def _pct(x) -> str:
    return f"{100 * x:.1f}"


def _ci(c) -> str:
    return f"{100 * c[0]:.1f}–{100 * c[1]:.1f}"


def _books(totals: dict) -> list[str]:
    return [b for b in ("GEN", "EPH", "ALL") if any(f"{b}.{t}" in totals for t in TRANSLATIONS)]


BOOK = {"GEN": "Genesis", "EPH": "Ephesians", "ALL": "All"}


def markdown(s: dict) -> str:
    BOOK["ALL"] = "All" if "Genesis" in s["label"] else s["label"]
    L = [f"# Word order (ORD) — {s['label']}", ""]
    if s["mock"]:
        L += ["> **MOCK — random judgements, layout test only. Not results.**", ""]
    it = s["items"]
    L += [f"ORD items: **{it['total']}** in {it['sentences']} sentences — "
          + ", ".join(f"{k} {v}" for k, v in sorted(it["by_role"].items(), key=lambda x: -x[1])) + ".", "",
          "An ORD item: a constituent (object, predicate, adverbial, prepositional phrase) that stands before "
          "the verb of its clause in the source (rules: order/items.py). **Sense**: does the English reader "
          "perceive the same prominence? **Source**: does the English keep the constituent before its verb?", ""]
    for d in R.DIMENSIONS:
        tot = s["ord"][d]["totals"]
        L += [f"## ORD alone — {R.DIM_LABEL[d]}", "",
              "| Book | Translation | Items | Fidelity F (95% CI) | Retention R | Distorted |", "|---|---|---|---|---|---|"]
        for b in _books(tot):
            for t in TRANSLATIONS:
                x = tot.get(f"{b}.{t}")
                if x:
                    L.append(f"| {BOOK[b]} | {t} | {x['features']} | **{_pct(x['fidelity'])}** ({_ci(x['fidelity_CI95'])}) | "
                             f"{_pct(x['retention'])} | {x['distorted']} |")
        L += ["", "Pairwise (Wilcoxon, Holm-corrected; r < 0 in “X–SPS” means SPS is higher):", ""]
        for b, c in s["ord"][d]["comparisons"].items():
            L.append(f"- {BOOK[b]} ({c['sentences']} sentences, Friedman p = {c['friedman_p']:.3g}): "
                     + "; ".join(f"{p['pair']} r = {p['r_rb']}, p = {p['p_holm']:.3g}" for p in c["pairwise"]))
        L.append("")
    L += ["## Each judge alone (F, all books)", "", "| Judge | Dimension | " + " | ".join(TRANSLATIONS) + " |",
          "|---|---|" + "---|" * len(TRANSLATIONS)]
    for j, dims in s["ord_per_judge"].items():
        for d, tot in dims.items():
            L.append(f"| {j} | {R.DIM_LABEL[d]} | " + " | ".join(
                _pct(tot[f'ALL.{t}']['fidelity']) if f"ALL.{t}" in tot else "—" for t in TRANSLATIONS) + " |")
    L += ["", "## Reliability", ""]
    for d, a in (s.get("judge_agreement") or {}).items():
        if a:
            L.append(f"- Agreement between the judges, {R.DIM_LABEL[d]}: {_pct(a['exact_agreement'])}% "
                     f"(α = {a['alpha']}, n = {a['n']})")
    for d, per in s["test_retest"].items():
        for j, a in per.items():
            if a:
                L.append(f"- Same judge asked again ({j}), {R.DIM_LABEL[d]}: {_pct(a['exact_agreement'])}% (n = {a['n']})")
    for j, c in s["source_crosscheck"].items():
        if c["cases"]:
            L.append(f"- Source verdicts vs. the position of the words in the English ({j}): "
                     f"{_pct(c['agreement'])}% agree (n = {c['cases']}; approximate check)")
    if "with_main" in s:
        L += ["", "## The main scores with ORD added", "",
              "| Book | Translation | Sense: main | Sense: main + ORD | Source: main | Source: main + ORD |",
              "|---|---|---|---|---|---|"]
        wm = s["with_main"]
        for b in _books(wm["sense"]["main"]):
            for t in TRANSLATIONS:
                k = f"{b}.{t}"
                if k in wm["sense"]["main"]:
                    L.append(f"| {BOOK[b]} | {t} | {_pct(wm['sense']['main'][k]['fidelity'])} | "
                             f"**{_pct(wm['sense']['main_plus_ord'][k]['fidelity'])}** | "
                             f"{_pct(wm['source']['main'][k]['fidelity'])} | "
                             f"**{_pct(wm['source']['main_plus_ord'][k]['fidelity'])}** |")
        for d in R.DIMENSIONS:
            for b, c in wm[d]["comparisons"].items():
                L.append(f"\n{R.DIM_LABEL[d]}, main + ORD, {BOOK[b]}: "
                         + "; ".join(f"{p['pair']} r = {p['r_rb']}, p = {p['p_holm']:.3g}" for p in c["pairwise"]))
    L += ["", "Every ORD verdict, with both judges' reasons: items.csv (contains the SPS text: keep it private).", ""]
    return "\n".join(L)


def main(argv: list[str]) -> None:
    ds = dataset("--john" in argv)
    use_ord(ds)
    build(ds, "--mock" in argv)


if __name__ == "__main__":
    main(sys.argv)
