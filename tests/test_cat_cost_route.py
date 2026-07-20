"""Integration tests for masked CAT costs built from trajectory tokens."""

from types import SimpleNamespace

import torch
from torch import nn

from trokens.models.cat_spatial_aggregation import CATSpatialCostAggregator
from trokens.models.pointformer import Pointformer


class _ConstantDenseCost(nn.Module):
    def __init__(self, value=0.25, input_resolution=(2, 2)):
        super().__init__()
        self.value = float(value)
        self.input_resolution = input_resolution
        self.calls = 0

    def forward_precomputed(
        self,
        correlation,
        dense_guidance,
        occupancy_mask,
    ):
        self.calls += 1
        del dense_guidance
        batch, temporal, num_labels, height, width = correlation.shape
        return correlation.new_full(
            (batch, temporal, num_labels, height, width),
            self.value,
        ) * occupancy_mask.unsqueeze(2)


def _stub_model(use_cost_agg=True, constant_cost=0.25):
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        MODEL=SimpleNamespace(FEAT_EXTRACT_MODE="nearest"),
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True),
    )
    model.pot_route_cfg = SimpleNamespace(
        FRAME_SOFTMAX_TAU=0.5,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    model.use_cat_cost_aggregation = use_cost_agg
    model.cat_spatial_cost_aggregator = _ConstantDenseCost(constant_cost)
    return model


def test_dense_cost_sampling_uses_track_coordinates_without_reshaping_points():
    model = _stub_model()
    dense_cost = torch.tensor(
        [[[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]]]
    )
    pred_tracks = torch.tensor(
        [[[[ -1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]]]
    )

    sampled = model._sample_dense_cost_at_tracks(dense_cost, pred_tracks)

    assert sampled.shape == (1, 2, 1, 4)
    assert torch.equal(sampled[0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(
        sampled[0, 1, 0],
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
    )


def test_dense_cost_sampling_keeps_fp32_track_coordinates_under_amp():
    model = _stub_model()
    dense_cost = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 1, 16)
    # For W=16 and align_corners=True this lies just inside cell 0. FP16
    # quantization moves it across the nearest-neighbor boundary into cell 1.
    pred_tracks = torch.tensor([[[[-0.93334, 0.0]]]], dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sampled = model._sample_dense_cost_at_tracks(dense_cost, pred_tracks)

    assert sampled.dtype == torch.float32
    assert sampled.item() == 0.0


def test_post_pointformer_cat_cost_directly_replaces_current_point_cosine():
    model = _stub_model(constant_cost=0.25)
    patch_tokens = torch.randn(2, 3, 4, 6)
    pred_tracks = torch.rand(2, 3, 4, 2) * 2.0 - 1.0
    point_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    text_features = torch.randn(5, 6)

    refined = model._compute_cat_refined_point_similarity(
        patch_tokens,
        point_mask,
        pred_tracks,
        text_features,
    )

    assert model.cat_spatial_cost_aggregator.calls == 1
    assert refined.shape == (2, 5, 3, 4)
    assert torch.equal(refined, torch.full_like(refined, 0.25))


def test_replacement_cost_is_not_clipped_to_cosine_range():
    model = _stub_model(constant_cost=2.0)
    model.cat_spatial_cost_aggregator.input_resolution = (1, 1)
    patch_tokens = torch.randn(1, 1, 1, 2)
    pred_tracks = torch.tensor([[[[0.0, 0.0]]]])
    point_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    text_features = torch.tensor([[1.0, 0.0]])

    refined = model._compute_cat_refined_point_similarity(
        patch_tokens,
        point_mask,
        pred_tracks,
        text_features,
    )

    assert refined.item() == 2.0


def test_rasterization_averages_collisions_and_keeps_empty_cells_blank():
    model = _stub_model()
    patch_tokens = torch.tensor(
        [[[[1.0, 3.0], [3.0, 5.0], [9.0, 7.0]]]]
    )
    point_similarity = torch.tensor(
        [[[[0.2, 0.6, 0.9]], [[-0.2, 0.2, 0.7]]]]
    )
    # First two trajectories collide in the top-left cell; bottom-right stays
    # empty because its only candidate is masked out.
    pred_tracks = torch.tensor(
        [[[[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]]]
    )
    point_mask = torch.tensor([[[True, True, False]]])

    dense_cost, dense_guidance, occupancy = (
        model._rasterize_point_cost_and_guidance(
            patch_tokens,
            point_similarity,
            pred_tracks,
            point_mask,
            resolution=(2, 2),
        )
    )

    assert torch.equal(
        occupancy,
        torch.tensor([[[[True, False], [False, False]]]]),
    )
    assert torch.allclose(dense_guidance[0, 0, 0, 0], torch.tensor([2.0, 4.0]))
    assert torch.allclose(dense_cost[0, 0, :, 0, 0], torch.tensor([0.4, 0.0]))
    assert torch.count_nonzero(dense_guidance[~occupancy]) == 0
    assert torch.count_nonzero(dense_cost.masked_select(~occupancy.unsqueeze(2))) == 0


def test_q2s_route_uses_replacement_cost_as_softmax_weights():
    value_tokens = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.9, 0.1], [0.1, 0.9]]],
            [[[0.1, 0.9], [0.9, 0.1]]],
        ]
    )
    pred_tracks = torch.tensor(
        [
            [[[-1.0, -1.0], [1.0, 1.0]]],
            [[[-1.0, -1.0], [1.0, 1.0]]],
            [[[-1.0, -1.0], [1.0, 1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 2, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, False, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": torch.tensor([4, 7]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [1, 0], [0, 1]],
            dtype=torch.bool,
        ),
    }
    text_features = torch.eye(2)

    model = _stub_model(use_cost_agg=True, constant_cost=0.25)
    model._get_pot_label_text_features = lambda class_ids, dtype: (
        text_features.to(dtype=dtype)
    )

    aux = model._build_frame_softmax_q2s_aux(
        value_tokens,
        metadata,
        pred_tracks=pred_tracks,
    )

    expected_query = torch.full((2, 2, 1, 2), 0.5)
    expected_support = torch.tensor([[[0.5, 0.5]], [[0.0, 0.0]]])
    assert torch.allclose(aux["query_partial_query_prototypes"], expected_query)
    assert torch.allclose(aux["query_partial_support_prototypes"], expected_support)


def test_real_cat_module_backpropagates_through_q2s_route():
    torch.manual_seed(7)
    model = _stub_model(use_cost_agg=True)
    model.cat_spatial_cost_aggregator = CATSpatialCostAggregator(
        appearance_dim=4,
        input_resolution=(4, 4),
        cost_dim=4,
        guidance_dim=2,
        num_heads=2,
        window_size=2,
        num_layers=1,
        mlp_ratio=2.0,
    )
    value_tokens = torch.randn(3, 2, 4, 4, requires_grad=True)
    pred_tracks = torch.rand(3, 2, 4, 2) * 2.0 - 1.0
    point_mask = torch.ones(3, 2, 4, dtype=torch.bool)
    metadata = {
        "support_mask": torch.tensor([True, False, False]),
        "pred_query_mask": point_mask,
        "pred_visibility": point_mask,
        "episode_class_ids": torch.tensor([4, 7]),
        "episode_positive_labels": torch.tensor(
            [[1, 0], [1, 0], [0, 1]],
            dtype=torch.bool,
        ),
    }
    text_features = torch.randn(2, 4)
    model._get_pot_label_text_features = lambda class_ids, dtype: (
        text_features.to(dtype=dtype)
    )

    aux = model._build_frame_softmax_q2s_aux(
        value_tokens,
        metadata,
        pred_tracks=pred_tracks,
    )
    loss = aux["query_partial_q2s_logits"].square().mean()
    loss.backward()

    assert value_tokens.grad is not None and torch.isfinite(value_tokens.grad).all()
    cat_gradients = [
        parameter.grad
        for parameter in model.cat_spatial_cost_aggregator.parameters()
    ]
    assert all(gradient is not None for gradient in cat_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in cat_gradients)
    assert sum(gradient.abs().sum() for gradient in cat_gradients) > 0
