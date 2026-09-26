"""Scores, statistics and the report: per sentence, then totals.

Feature score: retained 1 · partial 0.5 · lost 0 · distorted 0 · not_in_base excluded.
With two judges the feature score is the mean of the two (main results); each
judge's own results are reported too.

Per source sentence s and translation T:
    R  retention  = Σ score / scorable features             (loss = 1 − R)
    P  accuracy   = Σ score / (Σ score + distorted + unsupported additions)
    F  fidelity   = 2PR / (P + R)
Additions are judged per alignment group and shared among its sentences in
proportion to their feature counts.
Totals are feature-weighted (every feature of the book counts once).

Statistics: 95% bootstrap CI (sentences resampled, 2,000 draws); Friedman test
over the four translations on per-sentence F with Kendall's W; pairwise
Wilcoxon signed-rank tests, Holm-corrected, with matched-pairs rank-biserial r;
Krippendorff's α (nominal, outcomes) between judges and between runs.

    python -m sps_validation.report            # real results in build/judge/results/
    python -m sps_validation.report --mock     # pipeline test on random judgements (clearly labelled)
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from .judge import OUT, TRANSLATIONS, _jsonl, load_sources
from .validate import KINDS, find_rendering

SCORE = {"retained": 1.0, "partial": 0.5, "lost": 0.0, "distorted": 0.0}
CLASSES = ["LEX", "ASP", "STEM", "VOICE", "MOOD", "REF", "NUM", "DEF", "REL", "NEG", "ARG"]
JUDGES = ("claude", "gpt")
REPORT = Path("build/report")


# --------------------------------------------------------------------------- loading

def load_results(root: Path, set_name: str) -> dict[str, dict[str, dict]]:
    """{judge: {request id: record}} for judges that have results."""
    out = {}
    for j in JUDGES:
        p = root / j / f"{set_name}.jsonl"
        if p.exists():
            out[j] = {r["id"]: r for r in _jsonl(p)}
    return out


def feature_outcomes(record: dict) -> dict[str, dict]:
    res = record["result"]
    if res.get("status") not in ("ok", "incomplete"):
        return {}
    return {f["fid"]: f for f in res["features"]}


# --------------------------------------------------------------------------- per sentence

def sentence_rows(requests: list[dict], results: dict, units: dict, feats: dict) -> list[dict]:
    rows = []
    for req in requests:
        per_judge = {j: feature_outcomes(r[req["id"]]) for j, r in results.items() if req["id"] in r}
        if not any(per_judge.values()):
            continue
        adds = {j: [a for a in r[req["id"]]["result"].get("additions", []) if a["type"] == "unsupported"]
                for j, r in results.items() if req["id"] in r and r[req["id"]]["result"].get("status") in ("ok", "incomplete")}
        n_bead = len(req["fids"])
        for uid in req["units"]:
            fl = feats[uid]
            scores, cls_scores = [], defaultdict(list)
            counts = Counter()
            translit = Counter()
            for f in fl:
                # Only judges that actually scored this feature count; each gets weight
                # 1/(number of such judges), so scores and outcome counts use the same weights
                # and a refusal or a missing answer cannot shift the result.
                answers = [o for fo in per_judge.values()
                           if (o := fo.get(f["fid"])) and o["outcome"] in SCORE]
                if not answers:
                    continue
                vals = [SCORE[o["outcome"]] for o in answers]
                for o in answers:
                    counts[o["outcome"]] += 1 / len(answers)
                    if o.get("transliterated"):
                        translit[o["outcome"]] += 1 / len(answers)
                v = sum(vals) / len(vals)
                scores.append(v)
                cls_scores[f["class"]].append(v)
            if not scores:
                continue
            S, N = sum(scores), len(scores)
            U = np.mean([len(a) for a in adds.values()]) * len(fl) / max(n_bead, 1) if adds else 0.0
            D = counts["distorted"]
            R = S / N
            P = S / (S + D + U) if S + D + U else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            u = units[uid]
            rows.append({
                "book": req["book"], "sentence": uid, "translation": req["translation"],
                "source": u["text"], "english": req["english"], "group": "+".join(req["units"]),
                "features": N, "sum_score": round(S, 4), "retention": round(R, 4), "loss": round(1 - R, 4),
                "accuracy": round(P, 4), "fidelity": round(F, 4),
                "retained": round(counts["retained"], 2), "partial": round(counts["partial"], 2),
                "lost": round(counts["lost"], 2), "distorted": round(D, 2), "unsupported_additions": round(U, 3),
                "transliterated_features": round(sum(translit.values()), 2),
                "transliterated_retained": round(translit["retained"] + 0.5 * translit["partial"], 2),
                **{f"R_{c}": round(sum(v) / len(v), 4) if v else "" for c, v in
                   ((c, cls_scores.get(c, [])) for c in CLASSES)},
                **{f"n_{c}": len(cls_scores.get(c, [])) for c in CLASSES},
            })
    return rows


# --------------------------------------------------------------------------- totals & statistics

def totals(rows: list[dict]) -> dict:
    out = {}
    for book in ("GEN", "EPH", "ALL"):
        for t in TRANSLATIONS:
            rs = [r for r in rows if r["translation"] == t and (book == "ALL" or r["book"] == book)]
            if not rs:
                continue
            S = sum(r["sum_score"] for r in rs)
            N = sum(r["features"] for r in rs)
            D = sum(r["distorted"] for r in rs)
            U = sum(r["unsupported_additions"] for r in rs)
            R, P = S / N, S / (S + D + U)
            ci = bootstrap(rs)
            out[f"{book}.{t}"] = {
                "sentences": len(rs), "features": N, "retention": round(R, 4), "loss": round(1 - R, 4),
                "accuracy": round(P, 4), "fidelity": round(2 * P * R / (P + R), 4),
                "retention_CI95": ci["retention"], "fidelity_CI95": ci["fidelity"],
                "distorted": round(D, 1), "unsupported_additions": round(U, 1),
                "by_class": {c: round(sum(r[f"R_{c}"] * r[f"n_{c}"] for r in rs if r[f"n_{c}"]) /
                                      max(sum(r[f"n_{c}"] for r in rs), 1), 4)
                             for c in CLASSES if sum(r[f"n_{c}"] for r in rs)},
                "transliterated_features": round(sum(r["transliterated_features"] for r in rs), 1),
                "transliterated_score": round(sum(r["transliterated_retained"] for r in rs) /
                                              max(sum(r["transliterated_features"] for r in rs), 1e-9), 4)
                if sum(r["transliterated_features"] for r in rs) else None,
            }
    return out


def bootstrap(rs: list[dict], draws: int = 2000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    S = np.array([r["sum_score"] for r in rs]); N = np.array([r["features"] for r in rs])
    D = np.array([r["distorted"] for r in rs]); U = np.array([r["unsupported_additions"] for r in rs])
    idx = rng.integers(0, len(rs), size=(draws, len(rs)))
    s, n, d, u = S[idx].sum(1), N[idx].sum(1), D[idx].sum(1), U[idx].sum(1)
    R = s / n
    P = s / (s + d + u)
    F = 2 * P * R / (P + R)
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"retention": q(R), "fidelity": q(F)}


def comparisons(rows: list[dict], measure: str = "fidelity") -> dict:
    out = {}
    for book in ("GEN", "EPH", "ALL"):
        table = defaultdict(dict)
        for r in rows:
            if book == "ALL" or r["book"] == book:
                table[r["sentence"]][r["translation"]] = r[measure]
        complete = [v for v in table.values() if all(t in v for t in TRANSLATIONS)]
        if len(complete) < 5:
            continue
        data = np.array([[v[t] for t in TRANSLATIONS] for v in complete])
        chi, p = stats.friedmanchisquare(*data.T)
        W = chi / (len(complete) * (len(TRANSLATIONS) - 1))
        pairs = []
        for a, b in combinations(range(len(TRANSLATIONS)), 2):
            diff = data[:, a] - data[:, b]
            nz = diff[diff != 0]
            if len(nz) == 0:
                pairs.append({"pair": f"{TRANSLATIONS[a]}–{TRANSLATIONS[b]}", "p": 1.0, "r_rb": 0.0,
                              "median_diff": 0.0})
                continue
            w = stats.wilcoxon(data[:, a], data[:, b], zero_method="wilcox")
            ranks = stats.rankdata(np.abs(nz))
            rrb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
            pairs.append({"pair": f"{TRANSLATIONS[a]}–{TRANSLATIONS[b]}", "p": float(w.pvalue),
                          "r_rb": round(float(rrb), 3), "median_diff": round(float(np.median(diff)), 4)})
        order = sorted(range(len(pairs)), key=lambda i: pairs[i]["p"])
        running = 0.0
        for k, i in enumerate(order):
            running = max(running, min(1.0, pairs[i]["p"] * (len(pairs) - k)))
            pairs[i]["p_holm"] = running
        out[book] = {"sentences": len(complete), "friedman_chi2": round(float(chi), 3),
                     "friedman_p": float(p), "kendall_W": round(float(W), 4), "pairwise": pairs}
    return out


def krippendorff_nominal(pairs: list[tuple[str, str]]) -> float | None:
    """α for two coders, no missing values, nominal categories."""
    if not pairs:
        return None
    o = Counter()
    for a, b in pairs:
        o[(a, b)] += 1
        o[(b, a)] += 1
    n_c = Counter()
    for (a, _), v in o.items():
        n_c[a] += v
    n = sum(n_c.values())
    Do = sum(v for (a, b), v in o.items() if a != b) / n
    De = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n * (n - 1))
    return round(1 - Do / De, 4) if De else 1.0


def agreement(a: dict, b: dict, requests: list[dict], feats_by_fid: dict) -> dict:
    by_class = defaultdict(list)
    for req in requests:
        if req["id"] not in a or req["id"] not in b:
            continue
        fa, fb = feature_outcomes(a[req["id"]]), feature_outcomes(b[req["id"]])
        for fid in req["fids"]:
            if fid in fa and fid in fb:
                pair = (fa[fid]["outcome"], fb[fid]["outcome"])
                by_class[feats_by_fid[fid]["class"]].append(pair)
                by_class["ALL"].append(pair)
    return {c: {"n": len(p), "alpha": krippendorff_nominal(p),
                "exact_agreement": round(sum(x == y for x, y in p) / len(p), 4)}
            for c, p in by_class.items() if p}


def perturbation_summary(pert_reqs, pert_res, main_res) -> dict:
    """Sensitivity = planted error detected in the perturbed text; false-alarm rate = the
    same "detection" on the identical, unperturbed control judged in the same run.
    Only cases whose original (main run) target was judged retained are counted."""
    out = {}
    controls = {q["id"]: q for q in pert_reqs if q["perturbation"].get("control")}
    cases = [q for q in pert_reqs if not q["perturbation"].get("control")]
    unsupported = lambda r: sum(a["type"] == "unsupported" for a in r["result"].get("additions", []))
    for j in pert_res:
        cell = defaultdict(lambda: [0, 0, 0, 0])  # hits, cases, false alarms, controls
        collateral = [0, 0]
        for q in cases:
            ctrl_id = "c" + q["id"][1:]
            if q["id"] not in pert_res[j] or ctrl_id not in pert_res[j] or q["base_id"] not in main_res.get(j, {}):
                continue
            new, ctl, old = pert_res[j][q["id"]], pert_res[j][ctrl_id], main_res[j][q["base_id"]]
            if any(r["result"].get("status") != "ok" for r in (new, ctl, old)):
                continue
            kind, target = q["perturbation"]["kind"], q["perturbation"]["target"]
            if kind == "addition":
                hit = unsupported(new) > unsupported(old)
                alarm = unsupported(ctl) > unsupported(old)
            else:
                fn, fc, fo = feature_outcomes(new), feature_outcomes(ctl), feature_outcomes(old)
                if not all(target in x for x in (fn, fc, fo)) or fo[target]["outcome"] != "retained":
                    continue
                hit = SCORE.get(fn[target]["outcome"], 1) < 1
                alarm = SCORE.get(fc[target]["outcome"], 1) < 1
                for fid in fc:
                    if fid != target and fid in fn:
                        collateral[0] += fc[fid]["outcome"] != fn[fid]["outcome"]
                        collateral[1] += 1
            c = cell[(q["translation"], kind)]
            c[0] += hit
            c[1] += 1
            c[2] += alarm
            c[3] += 1

        def agg(keys):
            v = [sum(cell[k][i] for k in keys) for i in range(4)]
            return {"detected": v[0], "cases": v[1], "rate": round(v[0] / v[1], 3) if v[1] else None,
                    "false_alarms": v[2], "false_alarm_rate": round(v[2] / v[3], 3) if v[3] else None}

        out[j] = {
            "detection": {f"{t}.{k}": agg([(t, k)]) for (t, k) in sorted(cell)},
            "by_kind": {k: agg([x for x in cell if x[1] == k]) for k in KINDS},
            "by_translation": {t: agg([x for x in cell if x[0] == t]) for t in TRANSLATIONS},
            "collateral_change_rate": _rate(*collateral),
        }
    return out


def _rate(a, b):
    return {"detected": a, "cases": b, "rate": round(a / b, 3) if b else None}


def rule_crosscheck(requests, results, units, feats) -> dict:
    """NEG: negator present in the English (incl. neighbours) ⇔ judged retained/partial.
    NUM: English noun plural/singular matches the source number ⇔ judged retained/partial."""
    neg_re = re.compile(r"\b(not|no|never|none|nothing|nor|neither|without)\b|n[’']t\b", re.I)
    agree = defaultdict(lambda: [0, 0])
    for req in requests:
        toks = {t["id"]: t for u in req["units"] for t in units[u]["tokens"]}
        fs = {f["fid"]: f for u in req["units"] for f in feats[u]}
        for j, res in results.items():
            if req["id"] not in res:
                continue
            fo = feature_outcomes(res[req["id"]])
            for fid, o in fo.items():
                f = fs.get(fid)
                if not f or o["outcome"] not in SCORE:
                    continue
                judged = SCORE[o["outcome"]] > 0
                if f["class"] == "NEG":
                    rule = bool(neg_re.search(" ".join([req["before"], req["english"], req["after"]])))
                elif f["class"] == "NUM":
                    hit = find_rendering(req["english"], toks.get(f["token"], {}))
                    if not hit:
                        continue
                    plural = hit[2].lower().endswith("s") and not hit[2].lower().endswith("ss")
                    rule = plural == (f["value"] in ("plural", "dual"))
                else:
                    continue
                agree[(j, f["class"])][0] += rule == judged
                agree[(j, f["class"])][1] += 1
    return {f"{j}.{c}": _rate(*v) | {"label": "agreement"} for (j, c), v in agree.items()}


# --------------------------------------------------------------------------- mock (pipeline test only)

def write_mock(root: Path) -> None:
    rng = random.Random(1)
    outcomes = ["retained"] * 7 + ["partial", "lost", "distorted"]
    for set_name in ("main", "perturb", "retest"):
        reqs = _jsonl(OUT / "sets" / f"{set_name}.jsonl")
        for j in JUDGES:
            (root / j).mkdir(parents=True, exist_ok=True)
            with (root / j / f"{set_name}.jsonl").open("w", encoding="utf-8") as f:
                for r in reqs:
                    res = {"status": "ok", "missing": [],
                           "features": [{"fid": x, "outcome": rng.choice(outcomes), "english": "", "transliterated":
                                         r["translation"] == "SPS" and rng.random() < 0.05, "displaced": False,
                                         "reason": "MOCK"} for x in r["fids"]],
                           "additions": [{"english": "", "type": rng.choice(["grammatical", "unsupported"]),
                                          "reason": "MOCK"} for _ in range(rng.randint(0, 2))]}
                    meta = {k: r[k] for k in ("id", "translation", "book", "units", "perturbation", "base_id") if k in r}
                    f.write(json.dumps(meta | {"model": "MOCK", "result": res}) + "\n")


# --------------------------------------------------------------------------- output

def build(root: Path, out: Path, label: str) -> dict:
    units, feats = load_sources()
    feats_by_fid = {f["fid"]: f for fl in feats.values() for f in fl}
    main_reqs = _jsonl(OUT / "sets" / "main.jsonl")
    main = load_results(root, "main")
    if not main:
        raise SystemExit(f"no judge results in {root}")
    out.mkdir(parents=True, exist_ok=True)

    rows = restrict_to_sample(sentence_rows(main_reqs, main, units, feats))
    _write_csv(out / "sentences.csv", rows)
    served = {j: sorted({(x["result"].get("usage") or {}).get("served_model") or x["model"] for x in r.values()})
              for j, r in main.items()}
    summary = {"label": label, "judges": {j: ", ".join(v) for j, v in served.items()},
               "status": {j: dict(Counter(x["result"]["status"] for x in r.values())) for j, r in main.items()},
               "totals": totals(rows), "comparisons": comparisons(rows)}
    per_judge = {}
    for j in main:
        rj = restrict_to_sample(sentence_rows(main_reqs, {j: main[j]}, units, feats))
        per_judge[j] = {"totals": totals(rj), "comparisons": comparisons(rj)}
    summary["per_judge"] = per_judge
    if len(main) == 2:
        summary["judge_agreement"] = agreement(main["claude"], main["gpt"], main_reqs, feats_by_fid)
    retest = load_results(root, "retest")
    rt_reqs = {r["id"] for r in _jsonl(OUT / "sets" / "retest.jsonl")}
    summary["test_retest"] = {j: agreement(main[j], retest[j], [r for r in main_reqs if r["id"] in rt_reqs],
                                           feats_by_fid).get("ALL") for j in retest if j in main}
    pert = load_results(root, "perturb")
    if pert:
        summary["perturbation"] = perturbation_summary(_jsonl(OUT / "sets" / "perturb.jsonl"), pert, main)
    summary["rule_crosscheck"] = rule_crosscheck(main_reqs, main, units, feats)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    summary["unaligned_english"] = {
        t: {"pieces": sum(len(r.get("unaligned_english", [])) for r in main_reqs if r["translation"] == t),
            "words": sum(len(" ".join(r.get("unaligned_english", [])).split()) for r in main_reqs if r["translation"] == t)}
        for t in TRANSLATIONS}
    tables = article_tables(summary, rows)
    summary["article_tables"] = tables
    for name, t in tables.items():
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:  # -sig: opens cleanly in Excel
            w = csv.writer(f)
            w.writerow(t["columns"])
            w.writerows(t["rows"])
    (out / "report.md").write_text(tables_markdown(tables) + "\n" + markdown(summary), encoding="utf-8")
    return summary


def restrict_to_sample(rows: list[dict]) -> list[dict]:
    """With a sample (plan.py), only the sampled Genesis sentences and all of Ephesians count,
    so every translation is scored on exactly the same sentences."""
    path = OUT / "sample.json"
    if not path.exists():
        return rows
    keep = set(json.loads(path.read_text(encoding="utf-8"))["genesis_sentences"])
    return [r for r in rows if r["book"] == "EPH" or r["sentence"] in keep]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


GROUPS = {
    "Lexical meaning (semantic field)": ["LEX"],
    "Verbal grammar (aspect, stem, voice, mood)": ["ASP", "STEM", "VOICE", "MOOD"],
    "Reference and number (person, number, definiteness)": ["REF", "NUM", "DEF"],
    "Relations and syntax (relations, who-does-what, negation)": ["REL", "ARG", "NEG"],
}
BOOKS_LABEL = {"GEN": "Genesis", "EPH": "Ephesians", "ALL": "Both books"}


def article_tables(s: dict, rows: list[dict]) -> dict:
    """The four comparison tables for the article (also written as table1..4.csv)."""
    T = s["totals"]
    pct = lambda x: round(100 * x, 1)
    t1 = {"title": "Table 1. Overall fidelity to the source text (%, 95% confidence interval)",
          "columns": ["Book", "Translation", "Sentences", "Information items", "Retained (retention)",
                      "Accuracy", "Fidelity", "Fidelity 95% CI"], "rows": []}
    for b in ("GEN", "EPH", "ALL"):
        for t in TRANSLATIONS:
            x = T.get(f"{b}.{t}")
            if x:
                t1["rows"].append([BOOKS_LABEL[b], t, x["sentences"], x["features"], pct(x["retention"]),
                                   pct(x["accuracy"]), pct(x["fidelity"]),
                                   f"{pct(x['fidelity_CI95'][0])}–{pct(x['fidelity_CI95'][1])}"])
    t2 = {"title": "Table 2. Retention by type of information, both books (%)",
          "columns": ["Type of information"] + list(TRANSLATIONS), "rows": []}
    for label, classes in GROUPS.items():
        line = [label]
        for t in TRANSLATIONS:
            rs = [r for r in rows if r["translation"] == t]
            n = sum(r[f"n_{c}"] for r in rs for c in classes)
            v = sum(r[f"R_{c}"] * r[f"n_{c}"] for r in rs for c in classes if r[f"n_{c}"])
            line.append(pct(v / n) if n else "—")
        t2["rows"].append(line)
    t3 = {"title": "Table 3. What happens to the source information, both books",
          "columns": ["Translation", "Retained %", "Partly retained %", "Lost %", "Distorted %",
                      "Unsupported additions per 100 items"], "rows": []}
    for t in TRANSLATIONS:
        rs = [r for r in rows if r["translation"] == t]
        n = sum(r["features"] for r in rs)
        if n:
            t3["rows"].append([t] + [pct(sum(r[k] for r in rs) / n) for k in ("retained", "partial", "lost", "distorted")]
                              + [round(100 * sum(r["unsupported_additions"] for r in rs) / n, 2)])
    t4 = {"title": "Table 4. Are the differences real? Pairwise comparison of per-sentence fidelity",
          "columns": ["Book", "Pair", "Median difference", "Effect size (r)", "p (Holm)", "Significant (p < .05)"],
          "rows": [], "notes": []}
    for b in ("GEN", "EPH", "ALL"):
        c = s["comparisons"].get(b)
        if not c:
            continue
        t4["notes"].append(f"{BOOKS_LABEL[b]}: Friedman χ² = {c['friedman_chi2']}, p = {c['friedman_p']:.3g}, "
                           f"Kendall's W = {c['kendall_W']}, n = {c['sentences']} sentences.")
        for p in c["pairwise"]:
            t4["rows"].append([BOOKS_LABEL[b], p["pair"], p["median_diff"], p["r_rb"], f"{p['p_holm']:.3g}",
                               "yes" if p["p_holm"] < 0.05 else "no"])
    return {"table1": t1, "table2": t2, "table3": t3, "table4": t4}


def tables_markdown(tables: dict) -> str:
    L = ["# Article tables", ""]
    for t in tables.values():
        L += [f"**{t['title']}**", "", "| " + " | ".join(map(str, t["columns"])) + " |",
              "|" + "---|" * len(t["columns"])]
        L += ["| " + " | ".join(map(str, r)) + " |" for r in t["rows"]]
        L += [""] + [f"_{n}_" for n in t.get("notes", [])] + [""]
    return "\n".join(L)


def markdown(s: dict) -> str:
    L = [f"# Source-fidelity results{' — ' + s['label'] if s['label'] else ''}", "",
         "Judges: " + ", ".join(f"{j} = `{m}`" for j, m in s["judges"].items()), ""]
    for book, name in (("ALL", "All"), ("GEN", "Genesis"), ("EPH", "Ephesians")):
        L += [f"## Totals — {name}", "",
              "| Translation | Sentences | Features | Retention [95% CI] | Loss | Accuracy | Fidelity [95% CI] | Distorted | Unsupported additions |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in TRANSLATIONS:
            x = s["totals"].get(f"{book}.{t}")
            if x:
                L.append(f"| {t} | {x['sentences']} | {x['features']} | {x['retention']:.3f} "
                         f"[{x['retention_CI95'][0]:.3f}, {x['retention_CI95'][1]:.3f}] | {x['loss']:.3f} | "
                         f"{x['accuracy']:.3f} | {x['fidelity']:.3f} [{x['fidelity_CI95'][0]:.3f}, "
                         f"{x['fidelity_CI95'][1]:.3f}] | {x['distorted']} | {x['unsupported_additions']} |")
        L += ["", "Retention by feature class:", "",
              "| Translation | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for t in TRANSLATIONS:
            x = s["totals"].get(f"{book}.{t}")
            if x:
                L.append(f"| {t} | " + " | ".join(f"{x['by_class'][c]:.3f}" if c in x["by_class"] else "—"
                                                  for c in CLASSES) + " |")
        c = s["comparisons"].get(book)
        if c:
            L += ["", f"Friedman χ² = {c['friedman_chi2']}, p = {c['friedman_p']:.3g}, Kendall's W = "
                      f"{c['kendall_W']} (n = {c['sentences']} sentences).", "",
                  "| Pair | median diff (F) | rank-biserial r | p (Holm) |", "|---|---|---|---|"]
            for p in c["pairwise"]:
                L.append(f"| {p['pair']} | {p['median_diff']} | {p['r_rb']} | {p['p_holm']:.3g} |")
        L.append("")
    L += ["## Transliterated words", "", "| Translation | features on transliterated words | mean score |", "|---|---|---|"]
    for t in TRANSLATIONS:
        x = s["totals"].get(f"ALL.{t}")
        if x:
            L.append(f"| {t} | {x['transliterated_features']} | {x['transliterated_score'] if x['transliterated_score'] is not None else '—'} |")
    if s.get("unaligned_english"):
        L += ["", "English with no aligned source sentence (judged together with the preceding group):", ""]
        for t, v in s["unaligned_english"].items():
            L.append(f"- {t}: {v['pieces']} pieces, {v['words']} words")
    L += ["", "## Validity of the instrument", ""]
    if "judge_agreement" in s:
        L += ["Agreement between judges (Krippendorff's α, nominal):", "", "| Class | n | α | exact agreement |", "|---|---|---|---|"]
        for c, v in s["judge_agreement"].items():
            L.append(f"| {c} | {v['n']} | {v['alpha']} | {v['exact_agreement']} |")
        L.append("")
    if s.get("test_retest"):
        L += ["Test–retest (same judge, 10% of requests re-judged):", ""]
        for j, v in s["test_retest"].items():
            if v:
                L.append(f"- {j}: α = {v['alpha']}, exact agreement = {v['exact_agreement']} (n = {v['n']})")
        L.append("")
    for j, v in s.get("perturbation", {}).items():
        L += [f"Known-answer tests — {j} (detection rate of planted errors):", "",
              "| Error | " + " | ".join(TRANSLATIONS) + " | all |", "|---|" + "---|" * (len(TRANSLATIONS) + 1)]
        for k in KINDS:
            cells = [v["detection"].get(f"{t}.{k}", {}).get("rate") for t in TRANSLATIONS]
            L.append(f"| {k} | " + " | ".join("—" if c is None else f"{c:.2f}" for c in cells) +
                     f" | {v['by_kind'][k]['rate']} (false alarms {v['by_kind'][k]['false_alarm_rate']}) |")
        L += ["", "Sensitivity = planted error detected; false alarms = the same verdict on the identical "
                  "unperturbed control, judged in the same run.",
              f"Changes on untouched features between perturbed text and its control: "
              f"{v['collateral_change_rate']['rate']}", ""]
    if s.get("rule_crosscheck"):
        L += ["Rule cross-checks (agreement of judge with a deterministic rule):", ""]
        for k, v in s["rule_crosscheck"].items():
            L.append(f"- {k}: {v['rate']} (n = {v['cases']})")
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> None:
    if "--mock" in argv:
        root = OUT / "results_mock"
        write_mock(root)
        s = build(root, REPORT / "mock", "MOCK DATA — pipeline test, not a result")
    else:
        s = build(OUT / "results", REPORT, "")
    print(json.dumps({k: v for k, v in s["totals"].items() if k.startswith("ALL")}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
