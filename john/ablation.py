"""Ablations A1–A6 on John 1: the unchanged sps_validation.ablation, on the John 1 verdicts.

Nothing is re-judged and nothing is paid: every variant is recomputed from the categorical
verdicts already in build/john/judge/results/. The variants, their definitions and the
"conclusions changed" count are exactly those of the main study (docs/BASELINES.md §1); only the
inputs differ (one book, John 1, all 57 sentences, no sample).

    python -m john.ablation            # → build/john/report/ablation/
    python -m john.ablation --mock     # pipeline test on random judgements
"""

from __future__ import annotations

import sys

from sps_validation import ablation as A
from sps_validation import report as R

from .prepare import BUILD, CODE, load_units
from .report import LABEL, write_mock

JUDGE_DIR = BUILD / "judge"


def _comparisons(rows, measure: str = "fidelity") -> dict:
    """report.comparisons knows the books GEN/EPH and the pool ALL; for John 1 the pool is the book."""
    c = R.comparisons(rows, measure).get("ALL")
    return {CODE: c} if c else {}


def use_john() -> None:
    """Point the ablation module at John 1 (inputs and labels only; its functions are unchanged)."""
    A.OUT = JUDGE_DIR
    A.load_sources = load_units
    A.restrict_to_sample = lambda rows: rows  # John 1 is judged whole: no sample
    A.BOOKS = (CODE,)
    A.BOOK_LABEL = {CODE: LABEL}
    A.comparisons = _comparisons


def main(argv: list[str]) -> None:
    use_john()
    if "--mock" in argv:
        root = JUDGE_DIR / "results_mock"
        if not (root / "claude" / "main.jsonl").exists():
            write_mock(root)
        out = BUILD / "report_mock" / "ablation"
        res = A.run(root, out, "John 1 — MOCK DATA, pipeline test, not a result")
    else:
        out = BUILD / "report" / "ablation"
        res = A.run(JUDGE_DIR / "results", out, LABEL)
    print(f"{len(res['variants'])} variants; results in {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv)
