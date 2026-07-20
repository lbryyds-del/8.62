"""Tests for multi-label few-shot helpers shared across routing modes."""

import torch

from tools.few_shot_multilabel import (
    q2s_cos_sim_fp32,
    support_query_split_multilabel_conditioned,
)


def test_q2s_cosine_forces_fp32_for_large_tokens_and_keeps_gradients_finite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    x = (torch.rand(12, 1024, device=device, dtype=torch.float16) * 80).requires_grad_()
    y = (torch.rand(8, 1024, device=device, dtype=torch.float16) * 80).requires_grad_()
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        similarity = q2s_cos_sim_fp32(x, y)

    reference_numerator = x.float() @ y.float().transpose(-1, -2)
    reference_denominator = (
        torch.norm(x.float(), dim=-1, keepdim=True)
        @ torch.norm(y.float(), dim=-1, keepdim=True).transpose(-1, -2)
        + 0.01
    )
    reference = reference_numerator / reference_denominator
    assert similarity.dtype == torch.float32
    assert torch.isfinite(similarity).all()
    assert torch.allclose(similarity, reference, atol=1e-6, rtol=1e-5)

    similarity.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


def test_conditioned_split_replaces_support_and_query():
    base = {
        "support_preds": torch.randn(3, 2, 4, 8),
        "query_preds": torch.randn(1, 2, 4, 8),
        "query_condition": torch.tensor([False, False, True]),
    }
    query = torch.randn(1, 2, 1, 8)
    aux = {
        "support_conditioned_patch_tokens": torch.randn(2, 2, 1, 8),
        "support_branch_class_indices": torch.tensor([0, 1]),
        "query_conditioned_patch_tokens": query,
        "query_conditioned_sample_indices": torch.tensor([2]),
    }

    result = support_query_split_multilabel_conditioned(base, aux)

    assert result["support_preds"].shape == (3, 2, 1, 8)
    assert torch.equal(result["query_preds"], query)
