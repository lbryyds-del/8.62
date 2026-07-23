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
        self.last_text_features = None

    def forward_precomputed(
        self,
        correlation,
        dense_guidance,
        occupancy_mask,
        text_features=None,
    ):
        self.calls += 1
        self.last_text_features = text_features
        del dense_guidance
        batch, temporal, num_labels, height, width = correlation.shape
        return correlation.new_full(
            (batch, temporal, num_labels, height, width),
            self.value,
        ) * occupancy_mask.unsqueeze(2)


class _TextEncodedDenseCost(nn.Module):
    """Return each class text id as its dense cost and record label axes."""

    def __init__(self, input_resolution=(2, 2)):
        super().__init__()
        self.input_resolution = input_resolution
        self.text_axes = []

    def forward_precomputed(
        self,
        correlation,
        dense_guidance,
        occupancy_mask,
        text_features=None,
    ):
        del dense_guidance
        self.text_axes.append(text_features.detach().clone())
        batch, temporal, num_labels, height, width = correlation.shape
        class_ids = text_features[:, 0].reshape(1, 1, num_labels, 1, 1)
        return class_ids.expand(
            batch,
            temporal,
            num_labels,
            height,
            width,
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
    assert model.cat_spatial_cost_aggregator.last_text_features is text_features
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
    text_lookup = {4: text_features[0], 7: text_features[1]}

    model = _stub_model(use_cost_agg=True, constant_cost=0.25)
    model._get_pot_label_text_features = lambda class_ids, dtype: (
        torch.stack(
            [text_lookup[int(class_id)] for class_id in class_ids],
            dim=0,
        ).to(dtype=dtype)
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


def test_cat_uses_true_support_labels_and_episode_query_labels_without_leakage():
    model = _stub_model(use_cost_agg=True)
    model.cat_spatial_cost_aggregator = _TextEncodedDenseCost()
    model._get_pot_label_text_features = lambda class_ids, dtype: torch.stack(
        [
            torch.tensor(
                [float(class_id), 1.0],
                dtype=dtype,
                device=class_ids.device,
            )
            for class_id in class_ids
        ],
        dim=0,
    )

    value_tokens = torch.randn(3, 1, 4, 2)
    pred_tracks = torch.tensor(
        [
            [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]],
            [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]],
            [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    point_mask = torch.ones(3, 1, 4, dtype=torch.bool)
    support_mask = torch.tensor([True, False, False])
    episode_class_ids = torch.tensor([4, 7])
    episode_positive_labels = torch.tensor(
        [[True, False], [True, False], [False, True]]
    )
    raw_positive_labels = torch.zeros(3, 10, dtype=torch.bool)
    raw_positive_labels[0, [4, 9]] = True
    # These are query targets and must never affect the model-side label axis.
    raw_positive_labels[1, 1] = True
    raw_positive_labels[2, 2] = True
    episode_label_text = model._get_pot_label_text_features(
        episode_class_ids,
        value_tokens.dtype,
    )

    refined = model._compute_split_cat_refined_point_similarity(
        value_tokens,
        point_mask,
        pred_tracks,
        support_mask,
        episode_positive_labels,
        episode_class_ids,
        episode_label_text,
        raw_positive_labels=raw_positive_labels,
    )

    recorded_axes = model.cat_spatial_cost_aggregator.text_axes
    assert len(recorded_axes) == 2
    assert torch.equal(recorded_axes[0][:, 0], torch.tensor([4.0, 7.0]))
    assert torch.equal(recorded_axes[1][:, 0], torch.tensor([4.0, 9.0]))
    assert torch.equal(refined[0, 0], torch.full((1, 4), 4.0))
    assert torch.equal(refined[1:, 0], torch.full((2, 1, 4), 4.0))
    assert torch.equal(refined[1:, 1], torch.full((2, 1, 4), 7.0))

    changed_query_targets = raw_positive_labels.clone()
    changed_query_targets[1:] = ~changed_query_targets[1:]
    changed_episode_targets = episode_positive_labels.clone()
    changed_episode_targets[1:] = ~changed_episode_targets[1:]
    model.cat_spatial_cost_aggregator.text_axes.clear()
    changed = model._compute_split_cat_refined_point_similarity(
        value_tokens,
        point_mask,
        pred_tracks,
        support_mask,
        changed_episode_targets,
        episode_class_ids,
        episode_label_text,
        raw_positive_labels=changed_query_targets,
    )

    assert torch.equal(changed, refined)
    assert torch.equal(
        model.cat_spatial_cost_aggregator.text_axes[0][:, 0],
        torch.tensor([4.0, 7.0]),
    )
    assert torch.equal(
        model.cat_spatial_cost_aggregator.text_axes[1][:, 0],
        torch.tensor([4.0, 9.0]),
    )


def test_split_cat_accepts_fp16_inputs_with_fp32_refined_cost():
    model = _stub_model(use_cost_agg=True)
    model._get_pot_label_text_features = lambda class_ids, dtype: torch.ones(
        class_ids.numel(),
        2,
        dtype=dtype,
        device=class_ids.device,
    )

    def fp32_refined_cost(tokens, point_mask, pred_tracks, text_features):
        del point_mask, pred_tracks
        return torch.ones(
            tokens.shape[0],
            text_features.shape[0],
            tokens.shape[1],
            tokens.shape[2],
            dtype=torch.float32,
            device=tokens.device,
        )

    model._compute_cat_refined_point_similarity = fp32_refined_cost
    patch_tokens = torch.randn(2, 1, 2, 2, dtype=torch.float16)
    point_mask = torch.ones(2, 1, 2, dtype=torch.bool)
    pred_tracks = torch.zeros(2, 1, 2, 2)
    episode_class_ids = torch.tensor([4, 7])
    episode_label_text = torch.eye(2, dtype=torch.float16)

    refined = model._compute_split_cat_refined_point_similarity(
        patch_tokens,
        point_mask,
        pred_tracks,
        support_mask=torch.tensor([True, False]),
        episode_positive_labels=torch.tensor(
            [[True, False], [False, True]]
        ),
        episode_class_ids=episode_class_ids,
        episode_label_text=episode_label_text,
    )

    assert refined.dtype == torch.float32
    assert torch.equal(refined[1], torch.ones_like(refined[1]))
    assert torch.equal(refined[0, 0], torch.ones_like(refined[0, 0]))
    assert torch.count_nonzero(refined[0, 1]) == 0


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
        class_attention_enabled=True,
        class_guidance_dim=2,
        class_num_heads=2,
        class_attention_type="full",
        class_pooling_size=1,
        class_pad_len=0,
        class_gate_init=0.1,
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
    text_lookup = {4: text_features[0], 7: text_features[1]}
    model._get_pot_label_text_features = lambda class_ids, dtype: (
        torch.stack(
            [text_lookup[int(class_id)] for class_id in class_ids],
            dim=0,
        ).to(dtype=dtype)
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
