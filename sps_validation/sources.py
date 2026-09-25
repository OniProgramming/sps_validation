"""Fetch the Hebrew and Greek source datasets at pinned commits.

Every dataset is openly licensed. Pinning the commit makes the source side of
the study byte-for-byte reproducible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SOURCES = {
    # WLC text + OSHB morphology + Groves/Clear syntax trees + SDBH senses (CC BY 4.0)
    "macula-hebrew": {
        "url": "https://github.com/Clear-Bible/macula-hebrew.git",
        "commit": "47db250bd55d0d8577f2a94fba114ef16c35b23c",
        "paths": ["WLC/lowfat/01-Gen-*", "LICENSE.md"],
    },
    # SBLGNT text + morphology + syntax trees + Louw-Nida domains (CC BY 4.0)
    "macula-greek": {
        "url": "https://github.com/Clear-Bible/macula-greek.git",
        "commit": "8423afe47b9e8f24b7772e808af45c7159a6fe7e",
        "paths": ["SBLGNT/lowfat/10-ephesians.xml", "LICENSE.md"],
    },
    # Robinson-Pierpont Byzantine Textform, parsed (public domain)
    "byzantine-majority-text": {
        "url": "https://github.com/byztxt/byzantine-majority-text.git",
        "commit": "27a45ff1b7be6c17ccbfeac414f3f55732ae8e28",
        "paths": ["csv-unicode/strongs/with-parsing/EPH.csv", "LICENSE.txt"],
    },
    # Westcott-Hort, parsed by M. A. Robinson (public domain)
    "greektext-westcott-hort": {
        "url": "https://github.com/byztxt/greektext-westcott-hort.git",
        "commit": "91473892d7f36f1227e5c24f8d994d1da40311ad",
        "paths": ["parsed/EPH.UWH", "README.md"],
    },
}


def fetch(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        repo = dest / name
        if (repo / ".git").exists() and _head(repo) == spec["commit"]:
            print(f"{name}: present at {spec['commit'][:12]}")
            continue
        repo.mkdir(exist_ok=True)
        run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True)
        if not (repo / ".git").exists():
            run("init", "-q")
            run("remote", "add", "origin", spec["url"])
        run("sparse-checkout", "set", "--no-cone", *spec["paths"])
        run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", spec["commit"])
        run("checkout", "-q", spec["commit"])
        print(f"{name}: fetched {spec['commit'][:12]}")


def _head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip()


if __name__ == "__main__":
    fetch(Path(sys.argv[1] if len(sys.argv) > 1 else "data/sources"))
