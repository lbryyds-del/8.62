#!/usr/bin/env python3
"""Generate a point-tracking input CSV from hmdb_few_shot.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HMDB51 point-tracking csv with full video paths."
    )
    parser.add_argument(
        "--few-shot-csv",
        type=Path,
        required=True,
        help="Path to hmdb_few_shot.csv.",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        required=True,
        help="Root directory containing HMDB51 raw videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output csv path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="hmdb51",
        help="Dataset folder name to write into the csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = list(csv.DictReader(args.few_shot_csv.open("r", encoding="utf-8")))
    seen = set()
    tracking_rows = []
    videos_root = args.videos_root.resolve()

    for row in rows:
        vid_base_path = row["vid_base_path"]
        if vid_base_path in seen:
            continue
        seen.add(vid_base_path)
        tracking_rows.append(
            {
                "video_path": str((videos_root / vid_base_path).resolve()),
                "dataset": args.dataset_name,
                "vid_id": row["vid_id"],
                "class_name": row["class_name"],
                "label_id": row["label_id"],
                "split": row["split"],
                "num_frames": row["num_frames"],
            }
        )

    tracking_rows = sorted(tracking_rows, key=lambda row: row["video_path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "dataset",
                "vid_id",
                "class_name",
                "label_id",
                "split",
                "num_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(tracking_rows)

    print(f"Wrote {args.output}")
    print(f"video rows: {len(tracking_rows)}")


if __name__ == "__main__":
    main()
