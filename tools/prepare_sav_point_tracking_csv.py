#!/usr/bin/env python3
"""Generate a point-tracking input CSV from SAV few-shot text files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a SAV point-tracking csv with full video paths."
    )
    parser.add_argument(
        "--sav-root",
        type=Path,
        required=True,
        help="Root directory containing SAV videos and few-shot text files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output csv path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="sav",
        help="Dataset folder name to write into the csv.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional pbtxt label map. Defaults to SAV_ROOT/education_first_label.pbtxt.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "val", "test"],
        help="Few-shot split files to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of unique videos to write, useful for smoke tests.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write rows that have missing local videos instead of failing.",
    )
    return parser.parse_args()


def parse_line(line: str) -> tuple[str, Path]:
    """Return the source prefix and local relative video path."""
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    if "//" in line:
        prefix, rel_path = line.split("//", 1)
    else:
        prefix, rel_path = "", line
    return prefix, Path(rel_path)


def parse_label_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    items = re.findall(
        r'item\s*\{\s*name:\s*"([^"]+)"\s*id:\s*(\d+)\s*\}',
        text,
        re.S,
    )
    return {label_id: name for name, label_id in items}


def combo_to_names(combo: str, label_map: dict[str, str]) -> str:
    if not label_map:
        return ""
    return ";".join(label_map.get(label_id, label_id) for label_id in combo.split("_"))


def main() -> None:
    args = parse_args()
    sav_root = args.sav_root.resolve()
    label_map_path = args.label_map or sav_root / "education_first_label.pbtxt"
    label_map = parse_label_map(label_map_path)

    rows_by_path: dict[Path, dict[str, str]] = {}
    missing_paths: list[Path] = []
    total_rows = 0

    for split in args.splits:
        split_file = sav_root / f"{split}_few_shot.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        for line_no, line in enumerate(
            split_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            total_rows += 1
            prefix, rel_path = parse_line(line)
            parts = rel_path.parts
            if len(parts) < 3 or parts[0] != "videos":
                raise ValueError(
                    f"Unexpected SAV path format in {split_file}:{line_no}: {line}"
                )

            video_path = (sav_root / rel_path).resolve()
            if not video_path.exists():
                missing_paths.append(video_path)
                if not args.allow_missing:
                    continue

            class_name = parts[1]
            source_combo = prefix.removeprefix(split)
            key = video_path
            if key in rows_by_path:
                existing_splits = set(rows_by_path[key]["splits"].split(","))
                existing_splits.add(split)
                rows_by_path[key]["splits"] = ",".join(sorted(existing_splits))
                existing_prefixes = set(rows_by_path[key]["source_prefixes"].split(","))
                existing_prefixes.add(prefix)
                rows_by_path[key]["source_prefixes"] = ",".join(
                    sorted(existing_prefixes)
                )
                continue

            rows_by_path[key] = {
                "video_path": str(video_path),
                "dataset": args.dataset_name,
                "video_name": video_path.stem,
                "vid_id": str(rel_path),
                "class_name": class_name,
                "label_id": class_name,
                "label_names": combo_to_names(class_name, label_map),
                "splits": split,
                "source_prefixes": prefix,
                "source_combo": source_combo,
                "source_label_names": combo_to_names(source_combo, label_map),
            }

    if missing_paths and not args.allow_missing:
        sample = "\n".join(str(path) for path in missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} videos listed in SAV splits are missing. "
            f"First missing paths:\n{sample}"
        )

    rows = sorted(rows_by_path.values(), key=lambda row: row["video_path"])
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "dataset",
                "video_name",
                "vid_id",
                "class_name",
                "label_id",
                "label_names",
                "splits",
                "source_prefixes",
                "source_combo",
                "source_label_names",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print(f"input rows: {total_rows}")
    print(f"unique videos: {len(rows_by_path)}")
    print(f"written rows: {len(rows)}")
    print(f"missing videos: {len(missing_paths)}")


if __name__ == "__main__":
    main()
