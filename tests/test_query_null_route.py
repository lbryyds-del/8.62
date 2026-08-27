"""Tests for Query-only Null Evidence Routing (A2)."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.few_shot_multilabel import (
    get_query_null_orthogonal_loss,
    get_query_null_route_metrics,
)
from trokens.config.defaults import get_cfg
from trokens.models.pointformer import Pointformer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_score(score_init, score_min, score_max):
    ratio = (score_init - score_min) / (score_max - score_min)
    return math.log(ratio / (1.0 - ratio))


def _stub_model(
    tau=1.0,
    score_init=0.0,
    score_min=-1.0,
    score_max=1.0,
    cardinality_correction=True,
    value_scale=1.0,
    detach_frame_scale=True,
    detach_support=True,
    null_token=(0.0, 1.0),
):
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=tau,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.query_null_cfg = SimpleNamespace(
        SCORE_INIT=score_init,
        SCORE_MIN=score_min,
        SCORE_MAX=score_max,
        CARDINALITY_CORRECTION=cardinality_correction,
        TOKEN_INIT_STD=0.02,
        VALUE_SCALE=value_scale,
        DETACH_FRAME_SCALE=detach_frame_scale,
        ORTHO_WEIGHT=0.01,
        ORTHO_DETACH_SUPPORT=detach_support,
    )
    model.query_null_token = torch.nn.Parameter(
        torch.tensor([null_token], dtype=torch.float32)
    )
    model.query_null_score_raw = torch.nn.Parameter(torch.tensor(
        _raw_score(score_init, score_min, score_max),
        dtype=torch.float32,
    ))
    model.use_query_null_route = True
    model.use_support_text_fusion = False
    model.use_cat_cost_aggregation = False
    model.cfg = SimpleNamespace(
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
        MF=SimpleNamespace(POS_EMBED="joint"),
    )
    return model


def test_query_null_config_defaults_off_but_sav_enables_a2():
    cfg = get_cfg()
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ENABLE is False
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.SCORE_INIT == pytest.approx(0.07)
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ORTHO_WEIGHT == pytest.approx(0.01)

    cfg.merge_from_file(str(REPO_ROOT / "configs/trokens/sav.yaml"))
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.ENABLE is True
    assert cfg.FEW_SHOT.QUERY_NULL_ROUTE.SCORE_INIT == pytest.approx(0.07)
    assert cfg.FEW_SHOT.POT_ROUTE.ENABLE is True
    assert cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE is True
    assert cfg.FEW_SHOT.POT_ROUTE.FRAME_SOFTMAX_TAU == pytest.approx(0.04)


def test_query_null_score_is_bounded_and_parameters_skip_weight_decay():
    model = _stub_model(
        score_init=0.25,
        score_min=-0.2,
        score_max=0.8,
    )
    assert model._get_query_null_score().item() == pytest.approx(0.25)

    with torch.no_grad():
        model.query_null_score_raw.fill_(100.0)
    assert model._get_query_null_score().item() == pytest.approx(0.8)
    with torch.no_grad():
        model.query_null_score_raw.fill_(-100.0)
    assert model._get_query_null_score().item() == pytest.approx(-0.2)

    skip = model.no_weight_decay()
    assert "query_null_token" in skip
    assert "query_null_score_raw" in skip


@pytest.mark.parametrize("num_valid", [1, 2, 8])
def test_cardinality_correction_makes_equal_null_and_patch_evidence_half_mass(
    num_valid,
):
    model = _stub_model(cardinality_correction=True)
    total_points = num_valid + 2
    patch_tokens = torch.ones(1, total_points, 2)
    point_mask = torch.zeros(1, total_points, dtype=torch.bool)
    point_mask[:, :num_valid] = True
    similarity = torch.zeros(1, 1, total_points)

    _, patch_weights, null_weights = (
        model._compute_frame_softmax_query_prototypes_with_null_from_similarity(
            patch_tokens,
            point_mask,
            similarity,
        )
    )

    assert null_weights.item() == pytest.approx(0.5, abs=1e-6)
    assert patch_weights.sum().item() == pytest.approx(0.5, abs=1e-6)
    assert torch.equal(
        patch_weights[..., num_valid:],
        torch.zeros_like(patch_weights[..., num_valid:]),
    )


def test_without_cardinality_correction_null_has_one_over_n_plus_one_mass():
    model = _stub_model(cardinality_correction=False)
    num_points = 4
    patch_tokens = torch.ones(1, num_points, 2)
    point_mask = torch.ones(1, num_points, dtype=torch.bool)
    similarity = torch.zeros(1, 1, num_points)

    _, patch_weights, null_weights = (
        model._compute_frame_softmax_query_prototypes_with_null_from_similarity(
            patch_tokens,
            point_mask,
            similarity,
        )
    )

    assert null_weights.item() == pytest.approx(1.0 / 5.0, abs=1e-6)
    assert patch_weights.sum().item() == pytest.approx(4.0 / 5.0, abs=1e-6)


def test_null_mass_is_not_renormalized_and_value_uses_frame_token_norm():
    model = _stub_model(null_token=(0.0, 2.0))
    patch_tokens = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])
    point_mask = torch.ones(1, 2, dtype=torch.bool)
    similarity = torch.zeros(1, 1, 2)

    prototypes, patch_weights, null_weights = (
        model._compute_frame_softmax_query_prototypes_with_null_from_similarity(
            patch_tokens,
            point_mask,
            similarity,
        )
    )

    assert torch.allclose(patch_weights, torch.full_like(patch_weights, 0.25))
    assert torch.allclose(null_weights, torch.full_like(null_weights, 0.5))
    # Patch component: [.25*3, .25*4]; Null component: .5*mean(3,4)*[0,1].
    assert torch.allclose(prototypes, torch.tensor([[[0.75, 2.75]]]))


def test_all_masked_frame_routes_fully_to_unit_scale_null_value():
    model = _stub_model(null_token=(3.0, 4.0))
    patch_tokens = torch.tensor([[[8.0, 0.0], [0.0, 6.0]]])
    point_mask = torch.zeros(1, 2, dtype=torch.bool)
    similarity = torch.zeros(2, 1, 2)

    prototypes, patch_weights, null_weights = (
        model._compute_frame_softmax_query_prototypes_with_null_from_similarity(
            patch_tokens,
            point_mask,
            similarity,
        )
    )

    assert torch.equal(patch_weights, torch.zeros_like(patch_weights))
    assert torch.equal(null_weights, torch.ones_like(null_weights))
    expected = torch.tensor([0.6, 0.8]).view(1, 1, 2).expand(2, 1, 2)
    assert torch.allclose(prototypes, expected)


def test_query_null_route_is_amp_safe_and_both_parameters_receive_gradients():
    model = _stub_model(
        tau=0.4,
        score_init=0.2,
        null_token=(0.3, -0.7),
    )
    patch_tokens = torch.tensor(
        [[[1.0, 0.2], [0.1, 1.2], [-0.4, 0.7]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    point_mask = torch.tensor([[True, True, False]])
    text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prototypes, patch_weights, null_weights = (
            model._compute_frame_softmax_query_prototypes_with_null(
                patch_tokens,
                point_mask,
                text_features,
            )
        )
        loss = (
            prototypes.float().square().mean()
            + patch_weights.float().mean()
            + null_weights.float().mean()
        )
    loss.backward()

    assert torch.isfinite(prototypes.float()).all()
    assert torch.isfinite(patch_weights.float()).all()
    assert torch.isfinite(null_weights.float()).all()
    assert patch_tokens.grad is not None
    assert torch.isfinite(patch_tokens.grad.float()).all()
    assert model.query_null_token.grad is not None
    assert torch.isfinite(model.query_null_token.grad).all()
    assert model.query_null_score_raw.grad is not None
    assert torch.isfinite(model.query_null_score_raw.grad).all()


def test_orthogonal_loss_ignores_empty_support_and_detaches_support_by_default():
    model = _stub_model(
        detach_support=True,
        null_token=(1.0, 0.0),
    )
    support = torch.tensor(
        [[[1.0, 1.0], [0.0, 0.0]]],
        requires_grad=True,
    )

    loss = model._compute_query_null_orthogonal_loss(support)
    mean_abs, max_abs = model._compute_query_null_support_cosine_stats(support)
    loss.backward()

    expected_cosine = 2 ** -0.5
    assert loss.item() == pytest.approx(0.5, abs=1e-6)
    assert mean_abs.item() == pytest.approx(expected_cosine, abs=1e-6)
    assert max_abs.item() == pytest.approx(expected_cosine, abs=1e-6)
    assert support.grad is None
    assert model.query_null_token.grad is not None
    assert torch.isfinite(model.query_null_token.grad).all()


def test_orthogonal_loss_can_explicitly_update_support():
    model = _stub_model(
        detach_support=False,
        null_token=(1.0, 0.0),
    )
    support = torch.tensor([[[1.0, 1.0]]], requires_grad=True)

    model._compute_query_null_orthogonal_loss(support).backward()

    assert support.grad is not None
    assert torch.isfinite(support.grad).all()


def test_q2s_integration_changes_query_only_and_ignores_query_targets():
    model = _stub_model(tau=0.5, null_token=(1.0, -1.0))
    value_tokens = torch.tensor(
        [
            [[[1.0, 0.0], [0.8, 0.2]], [[1.0, 0.0], [0.9, 0.1]]],
            [[[0.0, 1.0], [0.2, 0.8]], [[0.0, 1.0], [0.1, 0.9]]],
            [[[0.7, 0.3], [0.3, 0.7]], [[0.6, 0.4], [0.4, 0.6]]],
        ]
    )
    point_mask = torch.ones(3, 2, 2, dtype=torch.bool)
    episode_text = torch.eye(2)
    model._get_pot_label_text_features = lambda class_ids, dtype: (
        episode_text.index_select(0, class_ids - 4).to(dtype=dtype)
    )
    metadata = {
        "support_mask": torch.tensor([True, True, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": torch.tensor([4, 5]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [0, 1], [1, 0]],
            dtype=torch.bool,
        ),
    }

    null_aux = model._build_frame_softmax_q2s_aux(value_tokens, metadata)
    assert null_aux["query_null_weights"].shape == (1, 2, 2)
    assert null_aux["query_partial_q2s_logits"].shape == (1, 2)
    assert torch.allclose(
        null_aux["query_partial_q2s_logits"],
        10.0 * null_aux["query_partial_diag_similarity"] - 2.0,
    )

    changed_metadata = dict(metadata)
    changed_metadata["episode_positive_labels"] = metadata[
        "episode_positive_labels"
    ].clone()
    changed_metadata["episode_positive_labels"][2] = torch.tensor([0, 1])
    changed_aux = model._build_frame_softmax_q2s_aux(
        value_tokens,
        changed_metadata,
    )
    assert torch.allclose(
        null_aux["query_partial_query_prototypes"],
        changed_aux["query_partial_query_prototypes"],
    )
    assert torch.allclose(
        null_aux["query_null_weights"],
        changed_aux["query_null_weights"],
    )

    model.use_query_null_route = False
    baseline_aux = model._build_frame_softmax_q2s_aux(value_tokens, metadata)
    assert "query_null_weights" not in baseline_aux
    assert torch.equal(
        null_aux["query_partial_support_prototypes"],
        baseline_aux["query_partial_support_prototypes"],
    )
    assert not torch.allclose(
        null_aux["query_partial_query_prototypes"],
        baseline_aux["query_partial_query_prototypes"],
    )


def test_query_null_loss_and_mechanism_metric_helpers():
    null_loss = torch.tensor(0.125, requires_grad=True)
    few_shot_aux = {
        "query_null_orthogonal_loss": null_loss,
        "query_null_weights": torch.tensor(
            [
                [[0.1, 0.3], [0.7, 0.9]],
                [[0.5, 0.7], [0.2, 0.4]],
            ]
        ),
        "query_null_score": torch.tensor(0.25),
        "query_null_support_mean_abs_cosine": torch.tensor(0.1),
        "query_null_support_max_abs_cosine": torch.tensor(0.3),
        "query_partial_diag_similarity": torch.tensor(
            [[0.8, 0.1], [0.2, 0.6]]
        ),
    }
    labels = torch.tensor([[1, 0], [0, 1]])

    extracted_loss = get_query_null_orthogonal_loss(
        few_shot_aux,
        torch.zeros(1),
    )
    metrics = get_query_null_route_metrics(few_shot_aux, labels)

    assert extracted_loss.item() == pytest.approx(null_loss.item())
    assert extracted_loss.requires_grad
    assert metrics["positive_null_mean"].item() == pytest.approx(0.25)
    assert metrics["negative_null_mean"].item() == pytest.approx(0.70)
    assert metrics["null_gap"].item() == pytest.approx(0.45)
    assert metrics["positive_diag_similarity"].item() == pytest.approx(0.70)
    assert metrics["negative_diag_similarity"].item() == pytest.approx(0.15)
    assert metrics["null_support_mean_abs_cosine"].item() == pytest.approx(0.1)
    assert metrics["null_support_max_abs_cosine"].item() == pytest.approx(0.3)
    assert all(not value.requires_grad for value in metrics.values())

    zero = get_query_null_orthogonal_loss(
        None,
        torch.zeros(1, dtype=torch.bfloat16),
    )
    assert zero.dtype == torch.float32
    assert zero.item() == 0.0
    assert get_query_null_route_metrics(None, labels) == {}
