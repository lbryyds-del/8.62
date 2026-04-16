#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SAV_ROOT="${SAV_ROOT:-$REPO_ROOT/data/sav}"
TROKENS_PT_DATA="${TROKENS_PT_DATA:-$REPO_ROOT/data/trokens_pt_data}"
TRACKING_CSV="${TRACKING_CSV:-$TROKENS_PT_DATA/few_shot_info/sav_point_tracking.csv}"
SHARD_DIR="${SHARD_DIR:-$TROKENS_PT_DATA/few_shot_info/sav_point_tracking_shards}"
RUN_NAME="${RUN_NAME:-sav_point_tracking_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/output/logs/$RUN_NAME}"
SPLITS="${SPLITS:-train test}"
GPUS="${GPUS:-0 1 2 3 4 5}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-bipartite}"
FPS="${FPS:-10}"
SEMANTIC_FEAT_EXTRACTOR="${SEMANTIC_FEAT_EXTRACTOR:-dino}"

read -r -a gpu_args <<< "$GPUS"
expanded_gpu_args=()
for gpu in "${gpu_args[@]}"; do
  for ((worker_id = 0; worker_id < WORKERS_PER_GPU; worker_id++)); do
    expanded_gpu_args+=("$gpu")
  done
done
gpu_args=("${expanded_gpu_args[@]}")
num_shards="${#gpu_args[@]}"
if [[ "$num_shards" -eq 0 ]]; then
  echo "No GPUs provided in GPUS." >&2
  exit 1
fi

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
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
fi

rm -rf "$SHARD_DIR"
"$PYTHON" "$REPO_ROOT/tools/split_point_tracking_csv.py" \
  --input "$TRACKING_CSV" \
  --output-dir "$SHARD_DIR" \
  --num-shards "$num_shards" \
  --prefix sav_point_tracking_shard

mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

pids=()
for shard_id in "${!gpu_args[@]}"; do
  gpu="${gpu_args[$shard_id]}"
  shard_csv="$SHARD_DIR/sav_point_tracking_shard_$(printf "%02d" "$shard_id").csv"
  log_file="$LOG_DIR/gpu_${gpu}_shard_$(printf "%02d" "$shard_id").log"

  track_args=(
    new_point_tracking.py
    --clustering_method "$CLUSTERING_METHOD"
    --csv_path "$shard_csv"
    --base_feat_path "$TROKENS_PT_DATA"
    --semantic_feat_extractor "$SEMANTIC_FEAT_EXTRACTOR"
    --continue_on_error
    --failure_csv "$LOG_DIR/failures_shard_$(printf "%02d" "$shard_id").csv"
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

  (
    cd "$REPO_ROOT/point_tracking"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "${track_args[@]}"
  ) > "$log_file" 2>&1 &
  pids+=("$!")
  echo "Started shard $shard_id on GPU $gpu, pid ${pids[-1]}, log $log_file"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "One or more point-tracking workers failed. Check logs in $LOG_DIR." >&2
  exit 1
fi

echo "All point-tracking workers completed."
