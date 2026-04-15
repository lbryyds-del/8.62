#!/usr/bin/env python3

"""Local HMDB51 train+test launcher for Trokens."""

import argparse
import os
import secrets
import string
import sys
from datetime import datetime
from pathlib import Path


def _default_run_name():
    return f"hmdb51_2gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _default_wandb_id():
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Launch HMDB51 training and testing with local defaults."
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        help="Physical GPU ids to expose, e.g. '0,1'.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29502")),
        help="Distributed master port.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", "16")),
        help="Data loader workers.",
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get("RUN_NAME") or _default_run_name(),
        help="Experiment name used for output and wandb.",
    )
    parser.add_argument(
        "--wandb-id",
        default=os.environ.get("WANDB_ID") or _default_wandb_id(),
        help="wandb run id.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default=os.environ.get("WANDB_MODE", "offline"),
        help="wandb mode for local runs.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "/data3/duanwei/data/hmdb51"),
        help="Path to HMDB51 videos.",
    )
    parser.add_argument(
        "--trokens-pt-data",
        default=os.environ.get(
            "TROKENS_PT_DATA", str(repo_root / "data/trokens_pt_data")
        ),
        help="Path to extracted point-tracking data.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Optional output directory. Defaults to output/hmdb/<run_name>.",
    )
    parser.add_argument(
        "--point-info-name",
        default=os.environ.get("POINT_INFO_NAME", "cotracker3_bip_fr_32"),
        help="Point-tracking folder name under TROKENS_PT_DATA.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "TROKENS_CONFIG", str(repo_root / "configs/trokens/hmdb.yaml")
        ),
        help="Config file to pass to tools/run_net.py.",
    )
    parser.add_argument(
        "--torch-home",
        default=os.environ.get("TORCH_HOME", str(repo_root / ".torch-cache")),
        help="Torch hub/cache root. Must contain the local hub repos expected by the model.",
    )
    parser.add_argument(
        "--train-enable",
        choices=("True", "False"),
        default="True",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--test-enable",
        choices=("True", "False"),
        default="True",
        help="Whether to run testing after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided via --gpus.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "output" / "hmdb" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["MASTER_PORT"] = str(args.master_port)
    env["NUM_WORKERS"] = str(args.num_workers)
    env["DATA_DIR"] = args.data_dir
    env["TROKENS_PT_DATA"] = args.trokens_pt_data
    env["RUN_NAME"] = args.run_name
    env["WANDB_ID"] = args.wandb_id
    env["WANDB_MODE"] = args.wandb_mode
    env["OUTPUT_DIR"] = str(output_dir)
    env["TORCH_HOME"] = args.torch_home

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(gpu_ids)}",
        f"--master_port={args.master_port}",
        "tools/run_net.py",
        "--init_method",
        "env://",
        "--new_dist_init",
        "--cfg",
        args.config,
        "WANDB.ID",
        args.wandb_id,
        "WANDB.EXP_NAME",
        args.run_name,
        "MASTER_PORT",
        str(args.master_port),
        "OUTPUT_DIR",
        str(output_dir),
        "NUM_GPUS",
        str(len(gpu_ids)),
        "DATA_LOADER.NUM_WORKERS",
        str(args.num_workers),
        "TRAIN.ENABLE",
        args.train_enable,
        "TEST.ENABLE",
        args.test_enable,
        "DATA.USE_RAND_AUGMENT",
        "True",
        "DATA.PATH_TO_DATA_DIR",
        args.data_dir,
        "DATA.PATH_TO_TROKEN_PT_DATA",
        args.trokens_pt_data,
        "FEW_SHOT.K_SHOT",
        "1",
        "FEW_SHOT.TRAIN_QUERY_PER_CLASS",
        "6",
        "FEW_SHOT.N_WAY",
        "5",
        "POINT_INFO.NAME",
        args.point_info_name,
        "POINT_INFO.SAMPLING_TYPE",
        "cluster_sample",
        "POINT_INFO.NUM_POINTS_TO_SAMPLE",
        "256",
        "MODEL.FEAT_EXTRACTOR",
        "dino",
        "MODEL.DINO_CONFIG",
        "dinov2_vitb14",
        "MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE",
        "True",
        "MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE",
        "True",
    ]

    print(f"Repo root: {repo_root}")
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"TORCH_HOME={env['TORCH_HOME']}")
    print("Launching:")
    print(" ".join(command))
    os.execvpe(sys.executable, command, env)


if __name__ == "__main__":
    main()
