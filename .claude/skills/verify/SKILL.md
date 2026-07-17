---
name: verify
summary: Run a minimal SAV few-shot episode through the real CLI.
---

# SAV runtime verification

Use an idle GPU and a temporary output directory. Point testing at the existing best checkpoint so temporary output does not need its own checkpoint.

```bash
python tools/run_local.py \
  --cfg configs/trokens/sav.yaml \
  --gpus 0 \
  --train-enable False \
  --test-enable True \
  --wandb-mode disabled \
  --num-workers 0 \
  --output-dir /tmp/trokens-verify \
  -- \
  TEST.CHECKPOINT_FILE_PATH "$PWD/output/sav/checkpoints/checkpoint_best.pyth" \
  FEW_SHOT.TEST_EPISODES 1
```

For the legacy route, append `FEW_SHOT.POT_ROUTE.MODE pot`. A successful run constructs the real SAV loader, loads the checkpoint, processes 10 examples, and prints base/novel/HM mAP.

Probe mode validation by appending `FEW_SHOT.POT_ROUTE.MODE invalid`; model construction should stop with a clear `Unsupported POT_ROUTE.MODE` error.
