import pytest
import torch

from trokens.models.cat_spatial_aggregation import CATSpatialCostAggregator


def _make_module(num_layers=1):
    return CATSpatialCostAggregator(
        appearance_dim=8,
        input_resolution=(8, 8),
        cost_dim=8,
        guidance_dim=4,
        num_heads=2,
        window_size=4,
        num_layers=num_layers,
        mlp_ratio=2.0,
        proj_dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
    )


def test_output_shape_and_fp32_cosine():
    torch.manual_seed(1)
    module = _make_module(num_layers=2)
    dense_patch = torch.randn(2, 3, 8, 8, 8)
    text_features = torch.randn(5, 8)

    refined_cost = module(dense_patch, text_features)

    assert refined_cost.shape == (2, 3, 5, 8, 8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        correlation = module.compute_correlation(
            dense_patch.to(torch.bfloat16),
            text_features.to(torch.bfloat16),
        )
    assert correlation.dtype == torch.float32
    assert correlation.shape == (2, 3, 5, 8, 8)
    assert torch.isfinite(correlation).all()


def test_frame_and_class_branches_are_isolated():
    torch.manual_seed(2)
    module = _make_module().eval()
    dense_patch = torch.randn(1, 3, 8, 8, 8)
    text_features = torch.randn(4, 8)
    reference = module(dense_patch, text_features)

    changed_frame = dense_patch.clone()
    changed_frame[:, 1] = torch.randn_like(changed_frame[:, 1]) * 7.0
    frame_output = module(changed_frame, text_features)
    assert torch.allclose(frame_output[:, 0], reference[:, 0], atol=1e-6, rtol=0.0)
    assert torch.allclose(frame_output[:, 2], reference[:, 2], atol=1e-6, rtol=0.0)
    assert not torch.allclose(frame_output[:, 1], reference[:, 1])

    changed_text = text_features.clone()
    changed_text[2] = torch.randn_like(changed_text[2]) * 7.0
    class_output = module(dense_patch, changed_text)
    keep_classes = torch.tensor([0, 1, 3])
    assert torch.allclose(
        class_output[:, :, keep_classes],
        reference[:, :, keep_classes],
        atol=1e-6,
        rtol=0.0,
    )
    assert not torch.allclose(class_output[:, :, 2], reference[:, :, 2])


def test_batched_videos_match_independent_execution():
    torch.manual_seed(3)
    module = _make_module().eval()
    dense_patch = torch.randn(2, 2, 8, 8, 8)
    text_features = torch.randn(3, 8)

    batched = module(dense_patch, text_features)
    independent = torch.cat(
        [module(dense_patch[index : index + 1], text_features) for index in range(2)],
        dim=0,
    )

    assert torch.allclose(batched, independent, atol=1e-5, rtol=1e-5)


def test_precomputed_cost_keeps_unoccupied_cells_blank_and_ignored():
    torch.manual_seed(5)
    module = _make_module().eval()
    correlation = torch.randn(1, 2, 3, 8, 8)
    guidance = torch.randn(1, 2, 8, 8, 8)
    occupancy = torch.zeros(1, 2, 8, 8, dtype=torch.bool)
    occupancy[:, :, 1:4, 2:6] = True
    occupancy[:, :, 6, 6] = True

    reference = module.forward_precomputed(
        correlation,
        guidance,
        occupancy,
    )

    changed_correlation = correlation.clone()
    changed_guidance = guidance.clone()
    changed_correlation.masked_fill_(~occupancy.unsqueeze(2), 1000.0)
    changed_guidance.masked_fill_(~occupancy.unsqueeze(-1), -1000.0)
    changed = module.forward_precomputed(
        changed_correlation,
        changed_guidance,
        occupancy,
    )

    expanded_mask = occupancy.unsqueeze(2).expand_as(reference)
    assert torch.count_nonzero(reference.masked_select(~expanded_mask)) == 0
    assert torch.allclose(changed, reference, atol=1e-6, rtol=0.0)


def test_gradients_are_finite_and_guidance_is_qk_only():
    torch.manual_seed(4)
    module = _make_module()
    dense_patch = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    text_features = torch.randn(3, 8, requires_grad=True)

    refined_cost = module(dense_patch, text_features)
    refined_cost.square().mean().backward()

    assert dense_patch.grad is not None
    assert text_features.grad is not None
    assert torch.isfinite(dense_patch.grad).all()
    assert torch.isfinite(text_features.grad).all()
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)

    attention = module.spatial_layers[0].window_attention.attention
    assert attention.q.in_features == module.cost_dim + module.guidance_dim
    assert attention.k.in_features == module.cost_dim + module.guidance_dim
    assert attention.v.in_features == module.cost_dim


def test_resolution_and_window_contract_is_validated():
    with pytest.raises(ValueError, match="must be divisible"):
        CATSpatialCostAggregator(
            appearance_dim=8,
            input_resolution=(10, 8),
            cost_dim=8,
            guidance_dim=4,
            num_heads=2,
            window_size=4,
        )

    module = _make_module()
    with pytest.raises(ValueError, match="does not match"):
        module(torch.randn(1, 2, 4, 8, 8), torch.randn(3, 8))
