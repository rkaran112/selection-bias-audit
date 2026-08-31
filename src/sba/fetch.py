"""Fetch the two LendingClub files.

    python -m sba.fetch

The dataset is "All Lending Club loan data" by Nathan George (wordsforthewise)
on Kaggle, released under CC0 (public domain). Kaggle's public download
endpoint serves it without credentials. Files land in data/raw/ and are
gitignored; the repo ships no data.

  accepted_2007_to_2018Q4.csv.gz   ~392 MB   2,260,701 funded loans
  rejected_2007_to_2018Q4.csv.gz   ~255 MB  27,648,741 declined applications
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from . import config as C

BASE = f"https://www.kaggle.com/api/v1/datasets/download/{C.KAGGLE_SLUG}"
FILES = [C.ACCEPTED_FILE, C.REJECTED_FILE]
MIN_BYTES = 100_000_000


def _progress(name: str):
    def hook(block, block_size, total):
        got = block * block_size
        if total > 0:
            pct = min(100.0, 100.0 * got / total)
            sys.stdout.write(f"\r  {name}  {pct:5.1f}%  "
                             f"{got / 1e6:7.1f} / {total / 1e6:.1f} MB")
        else:
            sys.stdout.write(f"\r  {name}  {got / 1e6:7.1f} MB")
        sys.stdout.flush()
    return hook


def fetch_one(name: str, force: bool = False) -> Path:
    dest = C.DATA_RAW / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > MIN_BYTES and not force:
        print(f"  {name}  already present "
              f"({dest.stat().st_size / 1e6:.0f} MB), skipping")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{BASE}/{name}"
    print(f"  downloading {name} ...")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress(name))
    sys.stdout.write("\n")
    if tmp.stat().st_size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name} downloaded only {tmp.stat().st_size} bytes. Kaggle may "
            f"have changed the endpoint; download it manually from "
            f"https://www.kaggle.com/datasets/{C.KAGGLE_SLUG} into data/raw/.")
    tmp.replace(dest)
    return dest


def main(force: bool = False) -> None:
    print(f"Fetching LendingClub data into {C.DATA_RAW}")
    for f in FILES:
        fetch_one(f, force)
    print("done. Now run:  python run_all.py")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
