#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEOS_ROOT="${VIDEOS_ROOT:-$REPO_ROOT/data/hmdb51/hmdb51}"
TROKENS_PT_DATA="${TROKENS_PT_DATA:-$REPO_ROOT/data/trokens_pt_data}"
FEW_SHOT_CSV="${FEW_SHOT_CSV:-$TROKENS_PT_DATA/few_shot_info/hmdb_few_shot.csv}"
TRACKING_CSV="${TRACKING_CSV:-$TROKENS_PT_DATA/few_shot_info/hmdb51_point_tracking.csv}"
SEMANTIC_FEAT_EXTRACTOR="${SEMANTIC_FEAT_EXTRACTOR:-dino}"

python3 "$REPO_ROOT/tools/prepare_hmdb_point_tracking_csv.py" \
  --few-shot-csv "$FEW_SHOT_CSV" \
  --videos-root "$VIDEOS_ROOT" \
  --output "$TRACKING_CSV"

cd "$REPO_ROOT/point_tracking"
python3 new_point_tracking.py \
  --clustering_method bipartite \
  --csv_path "$TRACKING_CSV" \
  --base_feat_path "$TROKENS_PT_DATA" \
  --semantic_feat_extractor "$SEMANTIC_FEAT_EXTRACTOR" \
  --fps 10
