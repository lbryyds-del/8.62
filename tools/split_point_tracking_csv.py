#!/usr/bin/env python3
"""Split a point-tracking CSV into round-robin shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a point-tracking CSV into balanced shard CSV files."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for shard CSV files."
    )
    parser.add_argument(
        "--num-shards", type=int, required=True, help="Number of shards to create."
    )
    parser.add_argument(
        "--prefix", default="shard", help="Output shard filename prefix."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    writers = []
    counts = [0 for _ in range(args.num_shards)]

    try:
        with args.input.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {args.input}")

            for shard_id in range(args.num_shards):
                shard_path = args.output_dir / f"{args.prefix}_{shard_id:02d}.csv"
                handle = shard_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
                writer.writeheader()
                handles.append(handle)
                writers.append(writer)

            for row_index, row in enumerate(reader):
                shard_id = row_index % args.num_shards
                writers[shard_id].writerow(row)
                counts[shard_id] += 1
    finally:
        for handle in handles:
            handle.close()

    for shard_id, count in enumerate(counts):
        shard_path = args.output_dir / f"{args.prefix}_{shard_id:02d}.csv"
        print(f"{shard_path}: {count}")


if __name__ == "__main__":
    main()
