"""Check that everything needed for the John 1 run is in place (writes nothing, sends nothing).

    python -m john.check
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from .prepare import INPUT_DIR


def main() -> None:
    ok = True

    def line(good: bool, text: str, fix: str = "") -> None:
        nonlocal ok
        ok &= good
        print(("OK      " if good else "MISSING ") + text + ("" if good or not fix else f"\n         → {fix}"))

    v = sys.version_info
    line((3, 10) <= v[:2] <= (3, 13), f"Python {v.major}.{v.minor}", "use Python 3.11")
    for pkg in ("anthropic", "openai", "numpy", "scipy"):
        try:
            importlib.import_module(pkg)
            line(True, f"package {pkg}")
        except ImportError:
            line(False, f"package {pkg}", "pip install -r requirements.txt")
    line(shutil.which("git") is not None, "git", "install Git for Windows: https://git-scm.com/download/win, "
                                                  "then close and reopen PyCharm")
    docx, txt = INPUT_DIR / "SPS_John1.docx", INPUT_DIR / "SPS_John1.txt"
    line(docx.exists() or txt.exists(), f"SPS text ({docx if docx.exists() else txt})",
         f"put SPS_John1.docx in {INPUT_DIR}")
    line((INPUT_DIR / "bsb.txt").exists(), f"BSB ({INPUT_DIR / 'bsb.txt'})",
         "download https://bereanbible.com/bsb.txt into that folder")
    line(Path("config/prices.json").exists(), "config/prices.json", "run the commands from the project folder")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        line(bool(os.environ.get(key)), f"{key} is set (value not shown)",
             f'in this terminal: $env:{key}="..."  (needed only for john.run)')
    print("\nAll set." if ok else "\nFix the MISSING lines, then run this check again.")


if __name__ == "__main__":
    main()
