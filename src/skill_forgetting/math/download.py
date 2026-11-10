#!/usr/bin/env python3
"""
Download supported math datasets into local per-type JSONL trees.

Supported outputs:
  - competition_math_splits/train/by_type/<type>.jsonl
  - hendrycks_math_splits/{train,test}/by_type/<type>.jsonl

Run with:
  python -m skill_forgetting.math.download --source competition_math
  python -m skill_forgetting.math.download --source hendrycks_math
"""
from __future__ import annotations

import argparse
import json
import io
import urllib.request
from pathlib import Path

ROOT = Path("data")
USE_PARQUET = False

TYPE_NAME_MAP = {
    "Algebra": "algebra",
    "Counting & Probability": "counting___probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}

HENDRYCKS_SUBSETS = {
    "algebra": "Algebra",
    "counting_and_probability": "Counting & Probability",
    "geometry": "Geometry",
    "intermediate_algebra": "Intermediate Algebra",
    "number_theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download supported math datasets into local per-type JSONL files.")
    parser.add_argument(
        "--source",
        choices=["competition_math", "hendrycks_math"],
        default="competition_math",
    )
    parser.add_argument("--output_root", type=Path, default=Path("data"))
    parser.add_argument("--parquet", action="store_true",
                        help="Read HF's native-split Parquet exports (no dataset loading script).")
    return parser.parse_args()


def write_rows(out_path: Path, rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_competition_math() -> None:
    from datasets import load_dataset

    out_dir = ROOT / "competition_math_splits" / "train" / "by_type"
    print("Loading qwedsacf/competition_math (train split)...")
    ds = load_dataset("qwedsacf/competition_math", split="train")
    print(f"  {len(ds)} examples, columns: {ds.column_names}")

    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, str]]] = {slug: [] for slug in TYPE_NAME_MAP.values()}
    skipped = 0
    for row in ds:
        slug = TYPE_NAME_MAP.get(row["type"])
        if slug is None:
            skipped += 1
            continue
        buckets[slug].append(
            {
                "problem": row["problem"],
                "level": row["level"],
                "type": row["type"],
                "solution": row["solution"],
            }
        )

    total = 0
    for slug, rows in sorted(buckets.items()):
        out_path = out_dir / f"{slug}.jsonl"
        write_rows(out_path, rows)
        print(f"  {slug:30s} {len(rows):>5d} examples  ->  {out_path}")
        total += len(rows)

    if skipped:
        print(f"  WARNING: skipped {skipped} rows with unrecognised type")
    print(f"  TOTAL {total} examples written to {out_dir}")


def download_hendrycks_math() -> None:
    if not USE_PARQUET:
        from datasets import load_dataset
    else:
        import pyarrow.parquet as pq

    out_root = ROOT / "hendrycks_math_splits"
    total_written = 0
    print("Loading EleutherAI/hendrycks_math by subset and native split...")
    for subset_name, display_type in HENDRYCKS_SUBSETS.items():
        slug = TYPE_NAME_MAP[display_type]
        for split_name in ("train", "test"):
            if USE_PARQUET:
                url = ("https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet/"
                       f"{subset_name}/{split_name}/0.parquet")
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = response.read()
                ds = pq.read_table(io.BytesIO(payload)).to_pylist()
            else:
                ds = load_dataset("EleutherAI/hendrycks_math", subset_name, split=split_name)
            rows = [
                {
                    "problem": row["problem"],
                    "level": row["level"],
                    "type": display_type,
                    "solution": row["solution"],
                }
                for row in ds
            ]
            out_path = out_root / split_name / "by_type" / f"{slug}.jsonl"
            write_rows(out_path, rows)
            total_written += len(rows)
            print(f"  {split_name:5s} {slug:30s} {len(rows):>5d} examples  ->  {out_path}")
    print(f"  TOTAL {total_written} examples written to {out_root}")


def main() -> None:
    global ROOT, USE_PARQUET
    args = parse_args()
    ROOT = args.output_root
    USE_PARQUET = args.parquet
    if args.source == "competition_math":
        download_competition_math()
    elif args.source == "hendrycks_math":
        download_hendrycks_math()


if __name__ == "__main__":
    main()
