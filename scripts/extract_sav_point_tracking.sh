#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SAV_ROOT="${SAV_ROOT:-$REPO_ROOT/data/sav}"
TROKENS_PT_DATA="${TROKENS_PT_DATA:-$REPO_ROOT/data/trokens_pt_data}"
TRACKING_CSV="${TRACKING_CSV:-$TROKENS_PT_DATA/few_shot_info/sav_point_tracking.csv}"
SPLITS="${SPLITS:-train test}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-bipartite}"
FPS="${FPS:-10}"
SEMANTIC_FEAT_EXTRACTOR="${SEMANTIC_FEAT_EXTRACTOR:-dino}"

prepare_args=(
  "$REPO_ROOT/tools/prepare_sav_point_tracking_csv.py"
  --sav-root "$SAV_ROOT"
  --output "$TRACKING_CSV"
  --splits
)
read -r -a split_args <<< "$SPLITS"
prepare_args+=("${split_args[@]}")
if [[ -n "${LIMIT:-}" ]]; then
  prepare_args+=(--limit "$LIMIT")
fi

"$PYTHON" "${prepare_args[@]}"

track_args=(
  "$REPO_ROOT/point_tracking/new_point_tracking.py"
  --clustering_method "$CLUSTERING_METHOD"
  --csv_path "$TRACKING_CSV"
  --base_feat_path "$TROKENS_PT_DATA"
  --semantic_feat_extractor "$SEMANTIC_FEAT_EXTRACTOR"
  --continue_on_error
  --failure_csv "$TROKENS_PT_DATA/few_shot_info/sav_point_tracking_failures.csv"
)
if [[ -n "$FPS" ]]; then
  track_args+=(--fps "$FPS")
fi
if [[ "${RERUN:-0}" == "1" ]]; then
  track_args+=(--rerun)
fi
if [[ "${MAKE_VIS:-0}" == "1" ]]; then
  track_args+=(--make_vis)
fi

cd "$REPO_ROOT/point_tracking"
"$PYTHON" "${track_args[@]}"
