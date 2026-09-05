from pathlib import Path

import numpy as np
import pytest

from trokens.config.defaults import get_cfg
from trokens.datasets.base_ds import _load_point_queries
from trokens.datasets.utils import _repair_decord_rgb_layout
from trokens.datasets.tinyvirat import (
    _split_tinyvirat_line,
    _tinyvirat_label_ids,
)


def test_tinyvirat_config_class_partition_matches_split_files():
    cfg = get_cfg()
    cfg.merge_from_file("configs/trokens/tinyvirat.yaml")

    assert cfg.MODEL.NUM_CLASSES == 21
    assert len(cfg.TRAIN.CLASS_NAME) == 12
    assert len(cfg.TEST.CLASS_NAME) == 21
    assert set(cfg.TEST.SEEN_LABELS).isdisjoint(cfg.TEST.NOVEL_LABELS)
    assert sorted(cfg.TEST.SEEN_LABELS + cfg.TEST.NOVEL_LABELS) == list(range(21))
    assert [cfg.TEST.CLASS_NAME[index] for index in cfg.TEST.SEEN_LABELS] == list(
        cfg.TRAIN.CLASS_NAME
    )

    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(cfg.TEST.CLASS_NAME)
    }
    expected_ids = {
        "train": set(cfg.TEST.SEEN_LABELS),
        "test": set(range(cfg.MODEL.NUM_CLASSES)),
    }
    for split, expected in expected_ids.items():
        actual = set()
        split_path = Path(cfg.DATA.PATH_TO_SPLIT_DIR) / f"{split}_few_shot.txt"
        for line in split_path.read_text(encoding="utf-8").splitlines():
            _, relative_path = _split_tinyvirat_line(line)
            class_combo = Path(relative_path).parts[1]
            actual.update(_tinyvirat_label_ids(class_combo, class_to_id))
        assert actual == expected


def test_tinyvirat_unknown_class_has_actionable_error():
    with pytest.raises(ValueError, match="TEST.CLASS_NAME"):
        _tinyvirat_label_ids("Opening-unknown_action", {"Opening": 0})


def test_point_queries_current_key_is_loaded():
    point_queries = _load_point_queries({"point_queries": [0, 3, 7]}, 3)

    assert point_queries.tolist() == [0, 3, 7]


def test_point_queries_count_must_match_tracks():
    with pytest.raises(ValueError, match="does not match pred_tracks"):
        _load_point_queries({"point_queries": [0, 3]}, 3)


@pytest.mark.parametrize("padded_channels", [4, 6])
def test_decord_row_padding_is_repaired_to_rgb(padded_channels):
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 3, 3)
    padded_rows = np.zeros((1, 2, 3 * padded_channels), dtype=np.uint8)
    padded_rows[..., : 3 * 3] = rgb.reshape(1, 2, 3 * 3)
    padded_video = padded_rows.reshape(1, 2, 3, padded_channels)

    repaired = _repair_decord_rgb_layout(padded_video)

    assert repaired.shape == rgb.shape
    np.testing.assert_array_equal(repaired, rgb)


def test_standard_decord_rgb_layout_is_unchanged():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 3, 3)

    assert _repair_decord_rgb_layout(rgb) is rgb
