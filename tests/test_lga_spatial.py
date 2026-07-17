from types import MethodType, SimpleNamespace

import torch

from trokens.models.pointformer import LGASpatialFineAttention, Pointformer
from tools.few_shot_multilabel import (
    q2s_cos_sim_fp32,
    support_query_split_multilabel_conditioned,
)


def _stub_model(mode="cluster", query_self_attn=True):
    model = Pointformer.__new__(Pointformer)
    torch.nn.Module.__init__(model)
    model.embed_dim = 8
    model.num_classes = 7
    model.lga_spatial_cfg = SimpleNamespace(
        INJECTION_MODE=mode,
        QUERY_SELF_ATTN=query_self_attn,
        CLUSTER_TAU=0.07,
        RESCALE_CLUSTER_SOFTMAX=True,
    )
    model.cfg = SimpleNamespace(
        POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=True)
    )
    model.lga_text_scale = torch.nn.Parameter(torch.tensor(1.0))
    model.lga_spatial_fine_attn = LGASpatialFineAttention(
        8, num_heads=2, mlp_ratio=2.0, attn_dropout=0.0, ffn_dropout=0.0
    )

    def fake_text(self, class_ids, dtype):
        features = torch.eye(8, device=class_ids.device, dtype=dtype)
        return features.index_select(0, class_ids)

    model._get_pot_label_text_features = MethodType(fake_text, model)
    return model


def _metadata(query_label=0.0):
    raw = torch.zeros(3, 7)
    raw[0, [1, 6]] = 1
    raw[1, 3] = 1
    raw[2, 5] = query_label
    episode = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [query_label, 1.0 - query_label, 0.0],
    ])
    return {
        "support_mask": torch.tensor([True, True, False]),
        "episode_positive_labels": episode,
        "episode_class_ids": torch.tensor([1, 3, 5]),
        "raw_positive_labels": raw,
        "pred_query_mask": torch.tensor([
            [[True, True, False], [True, True, False]],
            [[True, True, True], [False, False, False]],
            [[True, False, True], [True, False, True]],
        ]),
        "obj_ids": torch.tensor([[10, 20, 20], [4, 4, 9], [7, 8, 7]]),
    }


def test_fine_attention_shape_and_gradients():
    module = LGASpatialFineAttention(
        8, num_heads=2, mlp_ratio=2.0, attn_dropout=0.0, ffn_dropout=0.0
    )
    q = torch.randn(4, 5, 8, requires_grad=True)
    output = module(q, q, q, torch.zeros(4, 5, dtype=torch.bool))
    assert output.shape == q.shape
    output.square().mean().backward()
    assert q.grad is not None
    assert module.attn.in_proj_weight.grad is not None
    assert module.ffn[0].weight.grad is not None


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


def test_all_sav_labels_have_three_concise_prompts():
    model = Pointformer.__new__(Pointformer)
    label_names = [
        "sit", "stand", "look_forward", "look_sideways", "read",
        "flip_books", "touch_sth", "raise_hand", "hands_down",
        "take_notes", "applaud", "bend", "turn_around",
        "talk_with_others", "answer_questions",
    ]
    for label_name in label_names:
        prompts = model._get_sav_label_prompts(label_name)
        assert len(prompts) == 3
        assert all("student" not in prompt.lower() for prompt in prompts)


def test_cluster_and_global_track_weights():
    align = torch.zeros(2, 4, 8)
    align[:, :2, 0] = 1.0
    align[:, 2:, 1] = 1.0
    obj_ids = torch.tensor([11, 11, 42, 42])
    mask = torch.ones(2, 4, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0, 0, 0, 0, 0, 0, 0]])

    cluster_model = _stub_model("cluster")
    weights, cluster_ids, cluster_weights = cluster_model._compute_lga_track_weights(
        align, obj_ids, mask, target
    )
    assert cluster_ids.tolist() == [11, 42]
    assert weights[0, 0] > weights[0, 2]
    assert cluster_weights[0, 0] > cluster_weights[0, 1]

    global_model = _stub_model("global")
    weights, _, _ = global_model._compute_lga_track_weights(
        align, obj_ids, mask, target
    )
    assert torch.equal(weights, torch.ones_like(weights))


def test_equal_cluster_similarity_rescales_to_one():
    model = _stub_model("cluster")
    align = torch.zeros(1, 2, 8)
    align[0, 0, 0] = 1
    align[0, 1, 1] = 1
    _, _, cluster_weights = model._compute_lga_track_weights(
        align, torch.tensor([5, 9]), torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0.0, 0.0, 1.0, 0, 0, 0, 0, 0]]),
    )
    assert torch.allclose(cluster_weights, torch.ones_like(cluster_weights))


def test_episode_features_use_support_union_and_do_not_leak_query_labels():
    torch.manual_seed(4)
    model = _stub_model()
    model.eval()
    align = torch.randn(3, 2, 3, 8)
    visual = torch.randn(3, 2, 3, 8)
    first = model._build_lga_spatial_episode_features(
        align, visual, _metadata(query_label=0.0)
    )
    second = model._build_lga_spatial_episode_features(
        align, visual, _metadata(query_label=1.0)
    )
    assert first["support_conditioned_patch_tokens"].shape == (2, 2, 1, 8)
    assert first["support_branch_class_indices"].tolist() == [0, 1]
    assert first["support_branch_sample_indices"].tolist() == [0, 1]
    assert first["lga_text_class_ids"].tolist() == [1, 3, 5, 6]
    assert torch.allclose(
        first["query_conditioned_patch_tokens"],
        second["query_conditioned_patch_tokens"],
    )
    loss = first["support_conditioned_patch_tokens"].square().mean()
    loss.backward()
    assert not hasattr(model, "label_text_proj")
    assert model.lga_text_scale.grad is not None
    assert model.lga_spatial_fine_attn.attn.in_proj_weight.grad is not None


def test_invalid_tokens_do_not_change_masked_pooling():
    model = _stub_model()
    model.eval()
    metadata = _metadata()
    align = torch.randn(3, 2, 3, 8)
    visual = torch.randn(3, 2, 3, 8)
    first = model._build_lga_spatial_episode_features(align, visual, metadata)
    changed_align = align.clone()
    changed_visual = visual.clone()
    invalid = ~metadata["pred_query_mask"]
    changed_align[invalid] = 1e6
    changed_visual[invalid] = -1e6
    second = model._build_lga_spatial_episode_features(
        changed_align, changed_visual, metadata
    )
    assert torch.allclose(
        first["support_conditioned_patch_tokens"],
        second["support_conditioned_patch_tokens"], atol=1e-5,
    )
    assert torch.isfinite(second["support_conditioned_patch_tokens"]).all()


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
