"""Reproduce the metadata findings recorded in docs/DATA.md.

Pulls only the five small competition CSVs (~9 MB total) via the Kaggle API --
no DICOM pixel data -- then prints the statistics that DATA.md cites.

Usage:
    python scripts/explore_metadata.py [--meta-dir DIR] [--no-download]

Requires Kaggle API credentials (see README) and competition participation.
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

COMPETITION = "rsna-knee-abnormality-detection"
CSVS = ["train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv"]
META_COLS = ("StudyInstanceUID", "Report")


def download(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in CSVS:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION,
             "-f", name, "-p", str(meta_dir), "--force"],
            check=True,
        )


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def majority_script(text: str) -> str:
    """Classify a report by the alphabet most of its letters belong to."""
    names = ("CYRILLIC", "GREEK", "ARABIC", "HEBREW", "CJK", "HANGUL", "LATIN")
    counts: collections.Counter[str] = collections.Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts[next((n for n in names if n in name), "OTHER")] += 1
    return counts.most_common(1)[0][0] if counts else "EMPTY"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=Path("data/meta"))
    ap.add_argument("--no-download", action="store_true",
                    help="use CSVs already present in --meta-dir")
    args = ap.parse_args()

    if not args.no_download:
        download(args.meta_dir)

    train = pd.read_csv(args.meta_dir / "train.csv")
    series = pd.read_csv(args.meta_dir / "train_series.csv")
    test = pd.read_csv(args.meta_dir / "test.csv")
    test_series = pd.read_csv(args.meta_dir / "test_series.csv")
    submission = pd.read_csv(args.meta_dir / "sample_submission.csv")

    labels = [c for c in train.columns if c not in META_COLS]

    rule("1. FILES AND SHAPES")
    for name, df in [("train", train), ("train_series", series), ("test", test),
                     ("test_series", test_series), ("sample_submission", submission)]:
        print(f"{name:18s} {str(df.shape):>12s}")
    print(f"\nunique studies in train : {train.StudyInstanceUID.nunique()}")
    print(f"duplicate study rows    : {train.StudyInstanceUID.duplicated().sum()}")
    print(f"studies missing series  : {len(set(train.StudyInstanceUID) - set(series.StudyInstanceUID))}")
    print(f"\nlabels ({len(labels)}): {labels}")
    print(f"submission targets match train labels: {list(submission.columns[1:]) == labels}")
    print(f"Report column present at test time  : {'Report' in test.columns}")

    rule("2. LABEL AVAILABILITY")
    n_present = train[labels].notna().sum(axis=1)
    print("non-null labels per study:")
    print(n_present.value_counts().sort_index().to_string())
    gold = train[n_present == len(labels)]
    print(f"\nfully labeled : {len(gold)} ({100 * len(gold) / len(train):.1f}%)")
    print(f"unlabeled     : {int((n_present == 0).sum())}")

    rule("3. POSITIVE RATES AMONG THE GOLD SET (enriched -- not a prevalence estimate)")
    stats = sorted(((c, int((gold[c] == 1).sum())) for c in labels), key=lambda r: -r[1])
    for c, pos in stats:
        print(f"{c:18s} {pos:3d} pos {len(gold) - pos:3d} neg   {100 * pos / len(gold):5.1f}%")
    k = gold[labels].eq(1).sum(axis=1)
    print(f"\npositive findings per study: mean {k.mean():.2f}, range {k.min()}-{k.max()}")
    print(f"studies with zero findings : {int((k == 0).sum())}")

    rule("4. LABEL CO-OCCURRENCE (lift vs. independence, n=58 -- indicative only)")
    M = gold[labels].eq(1).astype(int)
    co = M.T @ M
    pairs = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            obs = int(co.loc[a, b])
            exp = M[a].sum() * M[b].sum() / len(M)
            pairs.append((a, b, obs, exp, obs / exp if exp else float("nan")))
    for a, b, obs, exp, lift in sorted(pairs, key=lambda p: -p[4])[:8]:
        print(f"{a:18s} + {b:18s} obs {obs:3d}  exp {exp:5.1f}  lift {lift:.2f}")

    rule("5. REPORTS")
    length = train.Report.fillna("").str.len()
    words = train.Report.fillna("").str.split().str.len()
    print(f"nulls: {train.Report.isna().sum()}   empty: {int((length == 0).sum())}")
    print("chars: min %d p25 %d median %d p75 %d p95 %d max %d"
          % (length.min(), length.quantile(.25), length.median(),
             length.quantile(.75), length.quantile(.95), length.max()))
    print("words: median %d p95 %d max %d"
          % (words.median(), words.quantile(.95), words.max()))
    print("\nmajority writing system:")
    print(train.Report.fillna("").map(majority_script).value_counts().to_string())

    rule("6. SERIES STRUCTURE")
    per_study = series.groupby("StudyInstanceUID").size()
    print("series per study: min %d median %.0f p95 %.0f max %d mean %.1f"
          % (per_study.min(), per_study.median(), per_study.quantile(.95),
             per_study.max(), per_study.mean()))
    print(per_study.value_counts().sort_index().to_string())
    print("\nplane counts:")
    print(series.Anatomical_Plane.value_counts().to_string())
    planes = series.groupby("StudyInstanceUID").Anatomical_Plane.nunique()
    print(f"\nstudies containing all three planes: {int((planes == 3).sum())} / {len(planes)}")

    print("\nFluid_Sensitive identical to Fat_Suppression?")
    print(f"  train: {(series.Fluid_Sensitive == series.Fat_Suppression).all()}")
    print(f"  test : {(test_series.Fluid_Sensitive == test_series.Fat_Suppression).all()}")
    print("\nplane x fluid-sensitive:")
    print(series.groupby(["Anatomical_Plane", "Fluid_Sensitive"]).size()
          .sort_values(ascending=False).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
