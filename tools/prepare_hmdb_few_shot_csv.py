#!/usr/bin/env python3
"""Generate class-disjoint HMDB51 few-shot CSV files for Trokens."""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hmdb_few_shot.csv from HMDB51 frame annotations and "
            "class split text files."
        )
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing hmdb51_train_split_*_frames.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the generated csv files will be written.",
    )
    parser.add_argument(
        "--split-id",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="HMDB51 frame-annotation split id used to recover video metadata.",
    )
    parser.add_argument(
        "--video-ext",
        default=".avi",
        help="Video extension to append to vid_base_path entries.",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=None,
        help="Optional root containing class subfolders with raw HMDB51 videos.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Optional root containing class subfolders with extracted frame directories.",
    )
    parser.add_argument(
        "--train-classes",
        type=Path,
        default=None,
        help="Optional path to the HMDB51 train/base class list.",
    )
    parser.add_argument(
        "--val-classes",
        type=Path,
        default=None,
        help="Optional path to the HMDB51 validation class list.",
    )
    parser.add_argument(
        "--test-classes",
        type=Path,
        default=None,
        help="Optional path to the HMDB51 test/novel class list.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    translated = (
        name.replace("!", "1")
        .replace("#", "")
        .replace("&", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(";", "")
        .replace("'", "")
        .replace(" ", "")
    )
    return translated.lower()


def tail_tokens(stem: str, n: int = 6) -> str:
    parts = stem.split("_")
    return "_".join(parts[-n:])


def build_video_index(videos_root: Path | None) -> dict[str, list[str]]:
    if videos_root is None:
        return {}
    video_index: dict[str, list[str]] = {}
    for class_dir in videos_root.iterdir():
        if class_dir.is_dir():
            video_index[class_dir.name] = sorted(p.stem for p in class_dir.glob("*.avi"))
    return video_index


def build_frame_count_index(frames_root: Path | None) -> dict[str, dict[str, int]]:
    if frames_root is None:
        return {}
    frame_index: dict[str, dict[str, int]] = {}
    for class_dir in frames_root.iterdir():
        if class_dir.is_dir():
            frame_index[class_dir.name] = {}
            for frame_dir in class_dir.iterdir():
                if frame_dir.is_dir():
                    frame_index[class_dir.name][frame_dir.name] = sum(
                        1 for _ in frame_dir.glob("*.jpg")
                    )
    return frame_index


def resolve_video_stem(
    ann_stem: str,
    class_name: str,
    num_frames: int,
    video_index: dict[str, list[str]],
    frame_count_index: dict[str, dict[str, int]],
) -> str | None:
    candidates = video_index.get(class_name, [])
    if not candidates:
        return ann_stem
    if ann_stem in candidates:
        return ann_stem

    ann_tail = tail_tokens(ann_stem)
    best_candidate = None
    best_score = -1.0
    for candidate in candidates:
        score = SequenceMatcher(
            None, normalize_name(ann_stem), normalize_name(candidate)
        ).ratio()
        if tail_tokens(candidate) == ann_tail:
            score += 10.0
        cand_frames = frame_count_index.get(class_name, {}).get(candidate)
        if cand_frames == num_frames:
            score += 5.0
        if score > best_score:
            best_candidate = candidate
            best_score = score
    if best_score >= 10.5:
        return best_candidate
    return None


def read_split_file(
    path: Path,
    split_name: str,
    video_ext: str,
    video_index: dict[str, list[str]],
    frame_count_index: dict[str, dict[str, int]],
    skipped_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            vid_rel_no_ext, num_frames, label_id = line.split()
            class_name, video_name = vid_rel_no_ext.split("/", 1)
            num_frames_int = int(num_frames)
            resolved_video_name = resolve_video_stem(
                ann_stem=video_name,
                class_name=class_name,
                num_frames=num_frames_int,
                video_index=video_index,
                frame_count_index=frame_count_index,
            )
            if resolved_video_name is None:
                skipped_rows.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "video_name_from_annotation": video_name,
                        "num_frames": num_frames_int,
                    }
                )
                continue
            rows.append(
                {
                    "vid_id": f"{class_name}/{resolved_video_name}",
                    "vid_base_path": f"{class_name}/{resolved_video_name}{video_ext}",
                    "class_name": class_name,
                    "video_name": resolved_video_name,
                    "label_id": int(label_id),
                    "num_frames": num_frames_int,
                    "split": split_name,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "vid_id",
        "vid_base_path",
        "class_name",
        "video_name",
        "label_id",
        "num_frames",
        "split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    annotations_dir: Path,
    split_id: int,
    video_ext: str,
    videos_root: Path | None,
    frames_root: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    full_path = annotations_dir / "hmdb51_full_frames.txt"
    train_path = annotations_dir / f"hmdb51_train_split_{split_id}_frames.txt"
    test_path = annotations_dir / f"hmdb51_test_split_{split_id}_frames.txt"
    if not full_path.exists() and not train_path.exists():
        raise FileNotFoundError(
            f"Missing HMDB51 metadata file: {full_path} or {train_path}"
        )
    if not full_path.exists() and not test_path.exists():
        raise FileNotFoundError(
            f"Missing HMDB51 metadata file: {full_path} or {test_path}"
        )

    video_index = build_video_index(videos_root)
    frame_count_index = build_frame_count_index(frames_root)
    skipped_rows: list[dict[str, object]] = []
    rows = []
    if full_path.exists():
        rows.extend(
            read_split_file(
                full_path,
                "all",
                video_ext,
                video_index,
                frame_count_index,
                skipped_rows,
            )
        )
        return rows, skipped_rows

    rows.extend(
        read_split_file(
            train_path,
            "train",
            video_ext,
            video_index,
            frame_count_index,
            skipped_rows,
        )
    )
    rows.extend(
        read_split_file(
            test_path,
            "test",
            video_ext,
            video_index,
            frame_count_index,
            skipped_rows,
        )
    )
    return rows, skipped_rows


def default_class_split_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    primary_paths = {
        "train": repo_root / "data/hmdb51/hmdb51_splits/base_classes.txt",
        "val": repo_root / "data/hmdb51/hmdb51_splits/val_classes.txt",
        "test": repo_root / "data/hmdb51/hmdb51_splits/novel_classes.txt",
    }
    if all(path.exists() for path in primary_paths.values()):
        return primary_paths

    fallback_paths = {
        "train": repo_root / "data/hmdb51/hmdb51_classes_train.txt",
        "val": repo_root / "data/hmdb51/hmdb51_classes_val.txt",
        "test": repo_root / "data/hmdb51/hmdb51_classes_test.txt",
    }
    if all(path.exists() for path in fallback_paths.values()):
        return fallback_paths

    missing = [
        str(path)
        for path in list(primary_paths.values()) + list(fallback_paths.values())
        if not path.exists()
    ]
    raise FileNotFoundError(
        "Could not find HMDB51 class split files. Checked:\n" + "\n".join(missing)
    )


def load_class_split_map(args: argparse.Namespace) -> dict[str, set[str]]:
    default_paths = default_class_split_paths()
    split_paths = {
        "train": args.train_classes or default_paths["train"],
        "val": args.val_classes or default_paths["val"],
        "test": args.test_classes or default_paths["test"],
    }

    class_splits: dict[str, set[str]] = {}
    for split_name, path in split_paths.items():
        with path.open("r", encoding="utf-8") as handle:
            class_names = {line.strip() for line in handle if line.strip()}
        if not class_names:
            raise ValueError(f"Class split file is empty: {path}")
        class_splits[split_name] = class_names

    overlaps = []
    split_names = ("train", "val", "test")
    for i, first in enumerate(split_names):
        for second in split_names[i + 1 :]:
            overlap = class_splits[first] & class_splits[second]
            if overlap:
                overlaps.append(
                    f"{first}/{second}: {len(overlap)} overlap -> {sorted(overlap)[:10]}"
                )
    if overlaps:
        raise ValueError(
            "Class split files must be class-disjoint, found overlaps:\n"
            + "\n".join(overlaps)
        )

    return class_splits


def dedupe_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    best_rows: dict[str, dict[str, object]] = {}
    duplicate_count = 0
    for row in rows:
        row_key = str(row["vid_id"])
        current_best = best_rows.get(row_key)
        if current_best is None:
            best_rows[row_key] = dict(row)
            continue

        duplicate_count += 1
        if int(row["num_frames"]) > int(current_best["num_frames"]):
            best_rows[row_key] = dict(row)

    return list(best_rows.values()), duplicate_count


def assign_class_splits(
    rows: list[dict[str, object]], class_splits: dict[str, set[str]]
) -> list[dict[str, object]]:
    split_by_class = {}
    for split_name, class_names in class_splits.items():
        for class_name in class_names:
            split_by_class[class_name] = split_name

    assigned_rows = []
    missing_classes = set()
    for row in rows:
        class_name = str(row["class_name"])
        split_name = split_by_class.get(class_name)
        if split_name is None:
            missing_classes.add(class_name)
            continue
        updated_row = dict(row)
        updated_row["split"] = split_name
        assigned_rows.append(updated_row)

    if missing_classes:
        raise ValueError(
            "Found classes without a split assignment: "
            + ", ".join(sorted(missing_classes))
        )

    present_classes = {str(row["class_name"]) for row in assigned_rows}
    expected_classes = set().union(*class_splits.values())
    absent_classes = expected_classes - present_classes
    if absent_classes:
        raise ValueError(
            "Class split file references classes with no videos in the generated rows: "
            + ", ".join(sorted(absent_classes))
        )

    return assigned_rows


def main() -> None:
    args = parse_args()
    rows, skipped_rows = build_rows(
        args.annotations_dir,
        args.split_id,
        args.video_ext,
        args.videos_root,
        args.frames_root,
    )
    class_splits = load_class_split_map(args)
    rows, duplicate_count = dedupe_rows(rows)
    rows = assign_class_splits(rows, class_splits)
    split_order = {"train": 0, "val": 1, "test": 2}
    rows = sorted(
        rows,
        key=lambda row: (
            split_order[str(row["split"])],
            int(row["label_id"]),
            str(row["vid_id"]),
        ),
    )

    split_path = args.output_dir / f"hmdb_few_shot_split{args.split_id}.csv"
    default_path = args.output_dir / "hmdb_few_shot.csv"

    write_csv(split_path, rows)
    write_csv(default_path, rows)

    train_count = sum(1 for row in rows if row["split"] == "train")
    val_count = sum(1 for row in rows if row["split"] == "val")
    test_count = sum(1 for row in rows if row["split"] == "test")
    print(f"Wrote {split_path}")
    print(f"Wrote {default_path}")
    print(f"train rows: {train_count}")
    print(f"val rows: {val_count}")
    print(f"test rows: {test_count}")
    print(f"deduped duplicates: {duplicate_count}")
    print(f"skipped rows: {len(skipped_rows)}")
    for row in skipped_rows[:20]:
        print(
            "SKIPPED",
            row["split"],
            row["class_name"],
            row["video_name_from_annotation"],
            row["num_frames"],
        )


if __name__ == "__main__":
    main()
