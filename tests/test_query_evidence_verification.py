"""Tests for Raw-Key evidence verification without Query patch rerouting."""

from types import SimpleNamespace

import pytest
import torch

from trokens.models.pointformer import Pointformer
from trokens.models.query_class_matchability import (
    build_query_evidence_map,
    classwise_frame_similarity,
    compute_evidence_conditioned_frame_matchability,
    confidence_aware_bimhm_logits,
)


def _pointformer(tau=1.0):
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.pot_route_cfg = SimpleNamespace(FRAME_SOFTMAX_TAU=float(tau))
    return model


def _evidence_cfg(**overrides):
    values = {
        "ENABLE": True,
        "MODE": "positive_confuser_margin",
        "EVIDENCE_SOURCE": "post",
        "LOG_PENALTY_WEIGHT": 0.0,
        "LOG_EPS": 0.05,
        "RELIABILITY_FALLBACK": False,
        "MARGIN_TEMPERATURE": 0.10,
        "MARGIN_BIAS": 0.0,
        "NEGATIVE_AGGREGATION": "max",
        "NEGATIVE_TOPK": 2,
        "NEGATIVE_TEMPERATURE": 0.10,
        "DETACH_CONFUSER_SUPPORT": True,
        "APPLY_DURING_TRAIN": True,
        "LOCAL_REFINEMENT_ENABLE": False,
        "EVIDENCE_VERIFICATION_ENABLE": True,
        "EVIDENCE_MAP_SOURCE": "raw",
        "EVIDENCE_MAP_TEMPERATURE": 0.07,
        "EVIDENCE_USE_VISIBILITY": True,
        "EVIDENCE_POSITIVE_AGGREGATION": "topk_mean",
        "EVIDENCE_POSITIVE_TOPK": 2,
        "EVIDENCE_CONFUSER_AGGREGATION": "topk_mean",
        "EVIDENCE_CONFUSER_TOPK": 2,
        "EVIDENCE_AGGREGATION_TEMPERATURE": 0.10,
        "EVIDENCE_DETACH_REFERENCES": True,
        "FRAME_MARGIN_TEMPERATURE": 0.10,
        "FRAME_MARGIN_BIAS": 0.0,
        "FRAME_LOG_PENALTY_WEIGHT": 0.10,
        "FRAME_LOG_EPS": 0.05,
        "FRAME_PENALTY_DIRECTION": "both",
        "EVIDENCE_VIDEO_TOPK_FRAMES": 3,
        "EVIDENCE_MIL_TEMPERATURE": 0.10,
        "EVIDENCE_MIL_LOSS_WEIGHT": 0.10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_raw_evidence_map_uses_pure_text_and_respects_mask():
    model = _pointformer()
    raw = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[True, True, False]]])
    result = build_query_evidence_map(
        model,
        raw,
        mask,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        temperature=0.05,
    )

    assert result["weights"].shape == (1, 2, 1, 3)
    assert result["weights"][0, 0, 0, 0] > 0.99
    assert result["weights"][0, 1, 0, 1] > 0.99
    assert torch.count_nonzero(result["weights"][..., 2]) == 0
    assert torch.allclose(
        result["weights"].sum(dim=-1),
        torch.ones(1, 2, 1),
    )


def test_frame_matchability_compares_identical_evidence_regions():
    weights = torch.tensor([[[[0.75, 0.25]]]])
    positive = torch.tensor([[[[0.8, 0.0]]]])
    confuser = torch.tensor([[[[0.0, 0.8]]]])
    result = compute_evidence_conditioned_frame_matchability(
        weights,
        positive,
        confuser,
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([1]),
        torch.tensor([1]),
        temperature=0.10,
    )

    assert result["positive_evidence"].item() == pytest.approx(0.6)
    assert result["confuser_evidence"].item() == pytest.approx(0.2)
    assert result["margin"].item() == pytest.approx(0.4)
    assert result["matchability"].item() > 0.98


def test_missing_confuser_or_frame_is_a_strict_noop():
    shape = (1, 1, 2, 2)
    result = compute_evidence_conditioned_frame_matchability(
        torch.full(shape, 0.5),
        torch.ones(shape),
        torch.zeros(shape),
        torch.tensor([[True, False]]),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    assert torch.equal(result["matchability"], torch.ones(1, 1, 2))
    assert torch.count_nonzero(result["margin"]) == 0
    assert not result["class_valid"].item()


def test_zero_frame_penalty_exactly_recovers_base_logits():
    similarity = torch.tensor([[[[0.9], [0.8]]]])
    rho = torch.tensor([[[0.01, 1.0]]])
    base = torch.tensor([[7.25]])
    result = confidence_aware_bimhm_logits(
        similarity,
        rho,
        base,
        alpha=10.0,
        penalty_weight=0.0,
    )
    assert torch.equal(result["logits"], base)
    assert torch.count_nonzero(result["frame_penalty"]) == 0


def test_frame_penalty_can_switch_only_support_to_query_winner():
    similarity = torch.tensor([[[[0.90], [0.88]]]])
    rho = torch.tensor([[[0.01, 1.0]]])
    base = torch.tensor([[6.9]])
    result = confidence_aware_bimhm_logits(
        similarity,
        rho,
        base,
        alpha=10.0,
        penalty_weight=0.10,
        eps=0.05,
    )

    assert result["base_winner"].item() == 0
    assert result["verified_winner"].item() == 1
    assert result["winner_switch_fraction"].item() == 1.0
    assert result["logits"].item() < base.item()


def test_bidirectional_frame_penalty_updates_both_bimhm_reductions():
    similarity = torch.tensor([[[[0.90], [0.80]]]])
    rho = torch.tensor([[[0.50, 1.00]]])
    base = torch.tensor([[7.25]])
    support_only = confidence_aware_bimhm_logits(
        similarity,
        rho,
        base,
        alpha=10.0,
        penalty_weight=0.10,
        eps=0.05,
        direction="support_to_query",
    )
    both = confidence_aware_bimhm_logits(
        similarity,
        rho,
        base,
        alpha=10.0,
        penalty_weight=0.10,
        eps=0.05,
        direction="both",
    )

    expected_penalty = 0.10 * torch.log(torch.tensor(0.50))
    assert both["q_to_s_delta"].item() == pytest.approx(
        expected_penalty.item() / 2.0,
        abs=1e-6,
    )
    assert both["s_to_q_delta"].item() == pytest.approx(
        expected_penalty.item(),
        abs=1e-6,
    )
    assert both["logits"].item() == pytest.approx(
        base.item() + 0.75 * expected_penalty.item(),
        abs=1e-6,
    )
    assert both["logits"].item() < support_only["logits"].item()


def test_frame_similarity_matches_manual_cosine_matrix():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    support = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    result = classwise_frame_similarity(query, support)
    assert result.shape == (1, 1, 2, 2)
    assert torch.equal(result, torch.eye(2).reshape(1, 1, 2, 2))


def test_wrapper_keeps_construction_route_and_ignores_query_targets():
    model = _pointformer(tau=1.0)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=_evidence_cfg(),
        ),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    model._get_pot_label_text_features = (
        lambda class_ids, dtype: torch.eye(2, dtype=dtype)
    )
    post = torch.tensor(
        [
            [[[1.0, 0.0], [0.8, 0.2]]],
            [[[0.0, 1.0], [0.2, 0.8]]],
            [[[0.9, 0.1], [0.1, 0.9]]],
        ],
        requires_grad=True,
    )
    raw = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    visibility = point_mask.clone()
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": visibility,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }
    changed = dict(metadata)
    changed["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed["episode_positive_labels"][-1] = torch.tensor([0, 1])

    # Exercise the actual meta-training path: APPLY_DURING_TRAIN=True keeps
    # both frame/global verification active, while Query targets remain loss-
    # only and cannot change routing outputs.
    model.train()
    first = model._build_frame_softmax_q2s_aux(
        post,
        metadata,
        matchability_evidence_tokens=raw,
    )
    second = model._build_frame_softmax_q2s_aux(
        post,
        changed,
        matchability_evidence_tokens=raw,
    )

    assert first["query_evidence_patch_weights"].shape == (1, 2, 1, 2)
    assert first["query_frame_matchability"].shape == (1, 2, 1)
    assert first["query_partial_q2s_temporal_logits"].shape == (1, 2)
    assert torch.equal(
        first["query_partial_query_prototypes"],
        first["query_partial_query_prototypes_before_refinement"],
    )
    for key in (
        "query_evidence_patch_weights",
        "query_frame_relative_margin",
        "query_partial_q2s_logits",
    ):
        assert torch.equal(first[key], second[key])


def test_evidence_and_local_refinement_are_mutually_exclusive():
    model = _pointformer(tau=1.0)
    model.cfg = SimpleNamespace(
        FEW_SHOT=SimpleNamespace(
            QUERY_CLASS_MATCHABILITY=_evidence_cfg(
                LOCAL_REFINEMENT_ENABLE=True,
            ),
        ),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_query_null_route = False
    model.use_cat_cost_aggregation = False
    model.use_support_text_fusion = False
    model._get_pot_label_text_features = (
        lambda class_ids, dtype: torch.eye(2, dtype=dtype)
    )
    values = torch.randn(3, 1, 2, 2)
    mask = torch.ones(3, 1, 2, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": mask,
        "pred_visibility": mask,
        "episode_class_ids": torch.tensor([0, 1]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }
    with pytest.raises(ValueError, match="controlled alternatives"):
        model._build_frame_softmax_q2s_aux(
            values,
            metadata,
            matchability_evidence_tokens=values,
        )
