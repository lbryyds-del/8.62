"""Tests for Support-calibrated Query-class matchability."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trokens.config.defaults import get_cfg
from trokens.models.query_class_matchability import (
    apply_log_matchability_penalty,
    compute_matchability_from_similarity,
    masked_topk_mean,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg(**overrides):
    values = {
        "TOPK_PATCHES": 1,
        "TOPK_FRAMES": 1,
        "CALIBRATION_BETA": 0.25,
        "TEMPERATURE": 0.05,
        "DETACH_SUPPORT_STATS": True,
        "LOG_PENALTY_WEIGHT": 0.25,
        "LOG_EPS": 0.05,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_enables_matchability_and_disables_learned_null():
    cfg = get_cfg()
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE is False
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ENABLE is False

    cfg.merge_from_file(str(REPO_ROOT / "configs/trokens/sav.yaml"))
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE is True
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ENABLE is False
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.TOPK_PATCHES == 8
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.TOPK_FRAMES == 3


def test_masked_topk_mean_ignores_invalid_entries():
    scores = torch.tensor([[0.9, 0.7, 100.0, -100.0]])
    mask = torch.tensor([[True, True, False, False]])
    value = masked_topk_mean(scores, mask, k=3)
    assert value.item() == pytest.approx(0.8)

    empty = masked_topk_mean(scores, torch.zeros_like(mask), k=2)
    assert empty.item() == 0.0


def test_support_calibration_separates_high_and_low_query_evidence():
    # First two rows are Support: class-0 positive/class-1 negative, then vice
    # versa. The third Query has strong class-0 and weak class-1 evidence.
    similarity = torch.tensor(
        [
            [[[[0.8]]], [[[0.2]]]],
            [[[[0.2]]], [[[0.8]]]],
            [[[[0.7]]], [[[0.1]]]],
        ]
    ).reshape(3, 2, 1, 1)
    point_mask = torch.ones(3, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor(
        [
            [1, 0],
            [0, 1],
            # Deliberately arbitrary Query targets; they are not consumed.
            [0, 1],
        ],
        dtype=torch.bool,
    )

    result = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels,
        _cfg(),
    )

    assert result["support_positive_evidence_mean"].tolist() == pytest.approx(
        [0.8, 0.8]
    )
    assert result["support_negative_evidence_mean"].tolist() == pytest.approx(
        [0.2, 0.2]
    )
    assert result["threshold"].tolist() == pytest.approx([0.35, 0.35])
    assert result["matchability"][0, 0].item() > 0.99
    assert result["matchability"][0, 1].item() < 0.01


def test_query_targets_do_not_change_matchability():
    similarity = torch.tensor(
        [
            [[[[0.8]]], [[[0.2]]]],
            [[[[0.2]]], [[[0.8]]]],
            [[[[0.6]]], [[[0.3]]]],
        ]
    ).reshape(3, 2, 1, 1)
    point_mask = torch.ones(3, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels_a = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
    labels_b = labels_a.clone()
    labels_b[-1] = torch.tensor([0, 1])

    result_a = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels_a,
        _cfg(),
    )
    result_b = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels_b,
        _cfg(),
    )
    assert torch.equal(result_a["matchability"], result_b["matchability"])
    assert torch.equal(result_a["threshold"], result_b["threshold"])


def test_log_matchability_is_only_a_penalty_and_is_bounded():
    base = torch.tensor([[2.0, -1.0]])
    matchability = torch.tensor([[1.0, 0.0]])
    final, penalty = apply_log_matchability_penalty(
        base,
        matchability,
        _cfg(LOG_PENALTY_WEIGHT=0.5, LOG_EPS=0.1),
    )

    assert penalty[0, 0].item() == pytest.approx(0.0)
    expected = 0.5 * torch.log(torch.tensor(0.1)).item()
    assert penalty[0, 1].item() == pytest.approx(expected)
    assert final[0, 0].item() == pytest.approx(base[0, 0].item())
    assert final[0, 1].item() < base[0, 1].item()
    assert torch.all(penalty <= 0.0)


def test_support_statistics_can_be_detached_without_blocking_query_gradient():
    similarity = torch.tensor(
        [
            [[[[0.8]]]],
            [[[[0.2]]]],
            [[[[0.5]]]],
        ],
        requires_grad=True,
    ).reshape(3, 1, 1, 1)
    point_mask = torch.ones(3, 1, 1, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False])
    labels = torch.tensor([[1], [0], [0]], dtype=torch.bool)

    result = compute_matchability_from_similarity(
        similarity,
        point_mask,
        support_mask,
        labels,
        _cfg(DETACH_SUPPORT_STATS=True),
    )
    result["matchability"].sum().backward()

    assert similarity.grad is not None
    # Support calibration is detached; only the Query row receives gradient.
    assert torch.equal(similarity.grad[:2], torch.zeros_like(similarity.grad[:2]))
    assert similarity.grad[2].abs().sum().item() > 0.0
