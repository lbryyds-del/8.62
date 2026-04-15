#!/usr/bin/env python3

"""SAV dataset."""

import os

import numpy as np
import pandas as pd

import trokens.utils.logging as logging

from .base_ds import BaseDataset
from .build import DATASET_REGISTRY


logger = logging.get_logger(__name__)


def _parse_label_combo(label_combo):
    """Parse SAV label combo strings like '1_3_10' into 0-based ids."""
    label_ids = []
    for label in str(label_combo).split("_"):
        label = label.strip()
        if label:
            label_ids.append(int(label) - 1)
    return label_ids


def _multi_hot(label_ids, num_classes):
    """Create a multi-hot vector for SAV atomic labels."""
    label_vec = np.zeros(num_classes, dtype=np.float32)
    for label_id in label_ids:
        if label_id < 0 or label_id >= num_classes:
            raise ValueError(
                f"SAV label id {label_id} is outside [0, {num_classes})."
            )
        label_vec[label_id] = 1.0
    return label_vec


@DATASET_REGISTRY.register()
class Sav(BaseDataset):
    """SAV dataset."""

    def __init__(self, cfg, mode):
        super(Sav, self).__init__(cfg, mode)

    def _construct_loader(self):
        """Construct the SAV video loader from sav_point_tracking.csv."""
        self.data_root = self.cfg.DATA.PATH_TO_DATA_DIR
        csv_name_to_use = self.cfg.DATA.DATA_CSV_NAME or "sav_point_tracking.csv"
        self.dataset_csv_path = os.path.join(self.splits_root, csv_name_to_use)
        self.dataset_df = pd.read_csv(self.dataset_csv_path)

        split_col = "splits" if "splits" in self.dataset_df.columns else "split"
        if "video_path" not in self.dataset_df.columns:
            self.dataset_df["video_path"] = self.dataset_df["vid_base_path"].apply(
                lambda x: os.path.join(self.data_root, x)
            )
        else:
            self.dataset_df["video_path"] = self.dataset_df["video_path"].apply(
                lambda x: x if os.path.isabs(str(x)) else os.path.join(self.data_root, str(x))
            )

        if "video_name" not in self.dataset_df.columns:
            self.dataset_df["video_name"] = self.dataset_df["video_path"].apply(
                lambda x: os.path.basename(x).split(".")[0]
            )
        if "vid_id" not in self.dataset_df.columns:
            self.dataset_df["vid_id"] = self.dataset_df["video_name"]

        self.dataset_df["feat_base_name"] = self.dataset_df["video_name"].apply(
            lambda x: str(x) + ".pkl"
        )

        self.split_df = self.dataset_df[
            self.dataset_df[split_col].astype(str).apply(
                lambda value: self.mode in {item.strip() for item in value.split(",")}
            )
        ].reset_index(drop=True)

        original_len = len(self.split_df)
        self.split_df["feat_path"] = self.split_df["feat_base_name"].apply(
            lambda x: os.path.join(self.base_feature_path, x)
        )
        self.split_df = self.split_df[
            self.split_df["video_path"].apply(os.path.exists)
        ].reset_index(drop=True)
        video_filtered_len = len(self.split_df)
        if video_filtered_len != original_len:
            logger.warning(
                "Filtered %s SAV rows with missing local videos for split %s.",
                original_len - video_filtered_len,
                self.mode,
            )

        self.split_df = self.split_df[
            self.split_df["feat_path"].apply(os.path.exists)
        ].reset_index(drop=True)
        new_len = len(self.split_df)
        if new_len != video_filtered_len:
            logger.warning(
                "Filtered %s SAV rows with missing point features for split %s.",
                video_filtered_len - new_len,
                self.mode,
            )

        if new_len == 0:
            raise FileNotFoundError(
                f"No SAV rows left for split {self.mode}. Check {self.dataset_csv_path} "
                f"and point features under {self.base_feature_path}."
            )

        if self.cfg.DATA.MULTI_LABEL:
            num_classes = self.cfg.MODEL.NUM_CLASSES
            self.split_df["label_combo"] = self.split_df["label_id"].astype(str)
            self.split_df["atomic_label_ids"] = self.split_df["label_combo"].apply(
                _parse_label_combo
            )
            self.split_df["label_id"] = self.split_df["atomic_label_ids"].apply(
                lambda labels: _multi_hot(labels, num_classes)
            )
            class_counts_all = {
                class_id: int(
                    self.split_df["atomic_label_ids"].apply(
                        lambda labels, cid=class_id: cid in labels
                    ).sum()
                )
                for class_id in range(num_classes)
            }
            class_counts = {
                class_id: count
                for class_id, count in class_counts_all.items()
                if count > 0
            }
            class_counts_to_check = list(class_counts.values())
        else:
            unique_labels = sorted(self.split_df["label_id"].astype(str).unique())
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.split_df["label_id"] = self.split_df["label_id"].astype(str).map(
                label_to_index
            )
            class_counts = self.split_df["label_id"].value_counts()
            class_counts_to_check = class_counts.tolist()

        samples_per_class = self.cfg.FEW_SHOT.K_SHOT + (
            self.cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS
            if self.mode == "train"
            else self.cfg.FEW_SHOT.TEST_QUERY_PER_CLASS
        )
        if len(class_counts) < self.cfg.FEW_SHOT.N_WAY:
            logger.warning(
                "SAV split %s has %s classes after filtering, fewer than FEW_SHOT.N_WAY=%s.",
                self.mode,
                len(class_counts),
                self.cfg.FEW_SHOT.N_WAY,
            )
        if any(count < samples_per_class for count in class_counts_to_check):
            logger.warning(
                "SAV split %s has classes with fewer than %s samples after filtering.",
                self.mode,
                samples_per_class,
            )

        self._path_to_videos = []
        self._make_final_lists()
        if self.cfg.DATA.MULTI_LABEL:
            self._atomic_labels_singles = self.split_df["atomic_label_ids"].tolist()
            self._atomic_labels = []
            for labels in self._atomic_labels_singles:
                for _ in range(self._num_clips):
                    self._atomic_labels.append(tuple(labels))
