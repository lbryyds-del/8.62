import json
from types import SimpleNamespace

import torch

from trokens.models.pointformer import Pointformer


class _ConstantCMWCostNet:
    def __init__(self, reliability=0.37, num_families=5):
        self.reliability = float(reliability)
        self.num_families = int(num_families)

    def __call__(
        self,
        token_evidence,
        label_context,
        point_mask,
        min_reliability=0.02,
    ):
        del label_context
        reliability = token_evidence.new_full(
            token_evidence.shape[:3],
            max(self.reliability, float(min_reliability)),
        )
        reliability = reliability * point_mask.unsqueeze(0).to(reliability.dtype)
        family_prob = token_evidence.new_full(
            (token_evidence.shape[0], self.num_families),
            1.0 / float(self.num_families),
        )
        candidate_reliability = reliability.unsqueeze(-1).expand(
            *reliability.shape,
            self.num_families,
        )
        return reliability, {
            "candidate_reliability": candidate_reliability,
            "family_prob": family_prob,
        }


def _model_with_psr_cfg():
    model = Pointformer.__new__(Pointformer)
    model.pot_route_cfg = SimpleNamespace(
        AFFINITY_TAU=0.07,
        ENTROPIC_EPS=0.01,
        MAX_ITERS=8,
        STOP_TOL=1e-5,
        UOT3D_ENTROPIC_EPS=0.03,
        UOT3D_MU_LOGIT_SCALE=5.0,
        UOT3D_RHO_FRAME=0.3,
        UOT3D_RHO_TRAJ=0.5,
        UOT3D_RHO_VIS=0.5,
        UOT3D_TARGET_MIX=0.85,
        UOT3D_TOTAL_MASS=1.0,
        UOT3D_TAU_FRAME=0.12,
        UOT3D_TAU_TRAJ=0.07,
        UOT3D_TAU_VIS=0.07,
        SHARED_TAU_LABEL=0.07,
        SHARED_THETA=0.2,
        SHARED_TAU_STRENGTH=0.1,
        UOT3D_SHARED_ENABLE=True,
        UOT3D_SHARED_RATIO=0.2,
        UOT3D_VIS_PRIVATE_WEIGHT=1.0,
        UOT3D_VIS_SHARED_WEIGHT=1.0,
        CMW_COST_NUM_FAMILIES=5,
        CMW_COST_HIDDEN_DIM=128,
        CMW_COST_MIN_RELIABILITY=0.02,
        CMW_COST_MARGIN_TAU=0.1,
        QUERY_PARTIAL_LABEL_CAP=1.0,
        QUERY_PARTIAL_VIS_CAP=1.0,
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BETA=1.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
        DEBUG_TOPK=3,
        DEBUG_SAVE_TOP_TOKENS=True,
    )
    model.cmw_cost_net = _ConstantCMWCostNet(reliability=0.37, num_families=5)
    return model


def _initial_shifted_kernel(model, cost, valid_mask, target_mass):
    valid3 = valid_mask.unsqueeze(0).expand_as(cost)
    valid_cost = cost.masked_fill(~valid3, 1e4)
    row_min = valid_cost.flatten(1).amin(dim=1).view(-1, 1, 1)
    row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
    initial = torch.exp(
        (-(valid_cost - row_min) / model.pot_route_cfg.UOT3D_ENTROPIC_EPS).clamp(
            min=-80.0,
            max=0.0,
        )
    )
    initial = torch.where(valid3, initial, torch.zeros_like(initial))
    return initial * (target_mass / initial.sum().clamp_min(1e-12))


def test_compute_sharedness_3d_reuses_entropy_strength_gate():
    model = _model_with_psr_cfg()
    sim = torch.tensor(
        [
            [[0.02, 0.90, 0.75, 0.80]],
            [[0.01, 0.10, 0.70, 0.78]],
            [[0.00, 0.00, 0.72, 0.79]],
        ],
        dtype=torch.float32,
    )
    point_mask = torch.tensor([[True, True, True, False]])

    sharedness, components = model._compute_sharedness_3d(
        sim,
        point_mask,
        return_components=True,
    )

    assert sharedness[0, 2] > sharedness[0, 0]
    assert sharedness[0, 2] > sharedness[0, 1]
    assert sharedness[0, 3] == 0.0
    assert components["semantic_strength"][0, 0] < components["semantic_strength"][0, 2]
    assert components["label_entropy"][0, 1] < components["label_entropy"][0, 2]


def test_psr_3d_uot_transport_shapes_and_soft_debug_are_stable():
    model = _model_with_psr_cfg()
    positive_text = torch.eye(4, dtype=torch.float32)[:3]
    st_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.8, 0.8, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.9, 0.1, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    point_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, True],
        ]
    )
    support_global = st_tokens[point_mask].mean(dim=0)

    out = model._compute_avg_3d_uot_transport(
        st_tokens,
        point_mask,
        support_global,
        positive_text,
        return_debug=True,
    )

    assert out["st_transport"].shape == (3, 2, 4)
    assert out["shared_transport"].shape == (3, 2, 4)
    assert out["cost_ext"].shape == (4, 2, 4)
    assert torch.isfinite(out["st_transport"]).all()
    assert torch.isfinite(out["shared_transport"]).all()
    assert out["st_transport"][:, 1, 2].sum() == 0.0
    assert out["shared_transport"][:, 1, 2].sum() == 0.0

    debug = out["debug"]
    assert debug["debug_type"] == "psr_3d_uot_soft"
    assert debug["config"]["shared_effective"] is True
    assert "transport_overlap" in debug
    assert "sharedness_summary" in debug
    target_debug = debug["targets"][0]
    assert "target_vs_shared_overlap" in target_debug
    assert "shared_absorption_ratio" in target_debug
    assert "frame_plane_l1_to_prior" in target_debug
    assert "traj_plane_l1_to_prior" in target_debug
    assert "vis_plane_l1_to_prior" in target_debug
    assert "target_frame_l1_to_prior" in target_debug
    assert "target_traj_l1_to_prior" in target_debug
    assert "shared_frame_l1_to_prior" in target_debug
    assert "shared_traj_l1_to_prior" in target_debug
    assert "frame_plane_cap_violation" not in target_debug
    json.dumps(debug)


def test_support_solver_uses_soft_priors_instead_of_caps():
    model = _model_with_psr_cfg()
    model.pot_route_cfg.MAX_ITERS = 32
    model.pot_route_cfg.UOT3D_RHO_VIS = 1.0
    cost = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.2, 0.1, 0.4], [0.3, 0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    prior_frame = torch.tensor([[0.35, 0.25], [0.20, 0.30]], dtype=torch.float32)
    prior_traj = torch.tensor(
        [[0.25, 0.30, 0.10], [0.20, 0.25, 0.20]],
        dtype=torch.float32,
    )
    prior_vis = torch.tensor(
        [[0.25, 0.20, 0.00], [0.20, 0.25, 0.15]],
        dtype=torch.float32,
    )

    gamma = model._solve_avg_3d_uot(
        cost,
        prior_frame,
        prior_traj,
        prior_vis,
        valid_mask,
    )
    initial = _initial_shifted_kernel(model, cost, valid_mask, prior_vis.sum())

    assert torch.isfinite(gamma).all()
    assert gamma[:, 0, 2].sum() == 0.0
    assert gamma.sum() > 0.0
    assert torch.abs(gamma.sum(dim=0) - prior_vis).sum() < torch.abs(
        initial.sum(dim=0) - prior_vis
    ).sum()

    loose_prior = torch.full_like(prior_vis, 10.0) * valid_mask.to(prior_vis.dtype)
    loose_gamma = model._solve_avg_3d_uot(
        cost,
        torch.full_like(prior_frame, 10.0),
        torch.full_like(prior_traj, 10.0),
        loose_prior,
        valid_mask,
    )
    loose_initial = _initial_shifted_kernel(model, cost, valid_mask, loose_prior.sum())
    assert not torch.allclose(loose_gamma, loose_initial, atol=1e-6)


def test_cmw_replaces_only_private_cost_rows():
    model = _model_with_psr_cfg()
    positive_text = torch.eye(4, dtype=torch.float32)[:3]
    st_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.8, 0.8, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.9, 0.1, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    point_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, True],
        ]
    )
    support_global = st_tokens[point_mask].mean(dim=0)

    out = model._compute_avg_3d_uot_transport(
        st_tokens,
        point_mask,
        support_global,
        positive_text,
        return_debug=True,
        target_label_indices=torch.tensor([0, 2]),
    )

    valid_private = point_mask.unsqueeze(0).expand(3, 2, 4)
    expected_private_cost = torch.full_like(out["private_cost"], 0.63)
    assert torch.allclose(
        out["private_cost"][valid_private],
        expected_private_cost[valid_private],
        atol=1e-6,
    )
    assert torch.allclose(
        out["cost"][valid_private],
        expected_private_cost[valid_private],
        atol=1e-6,
    )
    assert out["private_cost"][:, 1, 2].min() == 1e4
    assert out["cost_ext"][:3, 1, 2].min() == 1e4
    assert torch.allclose(
        out["cost_ext"][-1][point_mask],
        1.0 - out["sharedness"][point_mask],
        atol=1e-6,
    )
    assert out["st_transport"].shape == (2, 2, 4)
    assert out["target_label_indices"].tolist() == [0, 2]

    debug = out["debug"]
    assert debug["cost_source"] == "cmw_private_cost"
    assert "cmw_private_reliability_summary" in debug
    assert "cmw_target_reliability_summary" in debug["targets"][0]
    json.dumps(debug)


def test_cmw_raw_label_axis_keeps_target_subset_outputs():
    model = _model_with_psr_cfg()
    model.cfg = SimpleNamespace(POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=False))
    model.num_classes = 5
    model.atomic_label_names = [str(class_id) for class_id in range(5)]
    model._should_log_pot_debug = lambda: False
    model._get_pot_label_text_features = lambda ids, dtype: torch.nn.functional.one_hot(
        ids.cpu(),
        num_classes=5,
    ).to(dtype=dtype)

    calls = []

    def fake_uot(
        st_tokens,
        point_mask,
        support_global,
        positive_text,
        **kwargs,
    ):
        del st_tokens, support_global
        calls.append((positive_text.clone(), kwargs["target_label_indices"].clone()))
        num_targets = int(kwargs["target_label_indices"].numel())
        return {
            "st_transport": torch.ones(num_targets, *point_mask.shape),
            "sim": torch.zeros(positive_text.shape[0], *point_mask.shape),
        }

    model._compute_avg_3d_uot_transport = fake_uot
    value_tokens = torch.randn(2, 2, 3, 5)
    metadata = {
        "support_mask": torch.tensor([True, False]),
        "episode_positive_labels": torch.tensor(
            [[1, 0, 1], [0, 0, 0]],
            dtype=torch.float32,
        ),
        "raw_positive_labels": torch.tensor(
            [[1, 1, 0, 1, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
        ),
        "episode_class_ids": torch.tensor([0, 2, 3]),
        "pred_visibility": torch.ones(2, 2, 3, dtype=torch.bool),
    }

    aux = model._build_pot_support_prototypes(
        None,
        None,
        None,
        value_tokens,
        metadata,
    )
    positive_text, target_label_indices = calls[0]
    assert positive_text.shape[0] == 3
    assert target_label_indices.detach().cpu().tolist() == [0, 2]
    assert aux["support_branch_class_indices"].detach().cpu().tolist() == [0, 2]
    assert aux["support_conditioned_patch_tokens"].shape[0] == 2


def test_query_partial_solver_caps_marginals_without_amplifying_mass():
    model = _model_with_psr_cfg()
    cost = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.2, 0.1, 0.4], [0.3, 0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    frame_cap = torch.tensor([[0.35, 0.25], [0.20, 0.30]], dtype=torch.float32)
    traj_cap = torch.tensor(
        [[0.25, 0.30, 0.10], [0.20, 0.25, 0.20]],
        dtype=torch.float32,
    )
    vis_cap = torch.tensor(
        [[0.25, 0.20, 0.00], [0.20, 0.25, 0.15]],
        dtype=torch.float32,
    )

    gamma = model._solve_query_partial_3d_uot(
        cost,
        frame_cap,
        traj_cap,
        vis_cap,
        valid_mask,
    )

    tol = 1e-5
    assert torch.isfinite(gamma).all()
    assert gamma[:, 0, 2].sum() == 0.0
    assert torch.all(gamma.sum(dim=2) <= frame_cap + tol)
    assert torch.all(gamma.sum(dim=1) <= traj_cap + tol)
    assert torch.all(gamma.sum(dim=0) <= vis_cap + tol)

    loose_frame_cap = torch.full_like(frame_cap, 10.0)
    loose_traj_cap = torch.full_like(traj_cap, 10.0)
    loose_vis_cap = torch.full_like(vis_cap, 10.0)
    initial = torch.exp(
        (-cost / model.pot_route_cfg.UOT3D_ENTROPIC_EPS).clamp(min=-80.0, max=0.0)
    )
    initial = torch.where(valid_mask.unsqueeze(0), initial, torch.zeros_like(initial))
    unchanged = model._solve_query_partial_3d_uot(
        cost,
        loose_frame_cap,
        loose_traj_cap,
        loose_vis_cap,
        valid_mask,
    )
    assert torch.allclose(unchanged, initial, atol=1e-6)


def test_query_partial_batched_solver_matches_serial_targets():
    model = _model_with_psr_cfg()
    model.pot_route_cfg.MAX_ITERS = 16
    model.pot_route_cfg.STOP_TOL = 0.0
    cost = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.2, 0.1, 0.4], [0.3, 0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    frame_cap = torch.tensor([[0.35, 0.25], [0.20, 0.30]], dtype=torch.float32)
    traj_cap = torch.tensor(
        [[0.25, 0.30, 0.10], [0.20, 0.25, 0.20]],
        dtype=torch.float32,
    )
    vis_cap = torch.tensor(
        [[0.25, 0.20, 0.00], [0.20, 0.25, 0.15]],
        dtype=torch.float32,
    )

    cost_batch = torch.stack([cost, cost + 0.05], dim=0)
    frame_cap_batch = torch.stack([frame_cap, 0.8 * frame_cap], dim=0)
    traj_cap_batch = torch.stack([traj_cap, 0.8 * traj_cap], dim=0)
    vis_cap_batch = torch.stack([vis_cap, 0.8 * vis_cap], dim=0)

    batched = model._solve_query_partial_3d_uot(
        cost_batch,
        frame_cap_batch,
        traj_cap_batch,
        vis_cap_batch,
        valid_mask,
    )
    serial = torch.stack(
        [
            model._solve_query_partial_3d_uot(
                cost_batch[idx],
                frame_cap_batch[idx],
                traj_cap_batch[idx],
                vis_cap_batch[idx],
                valid_mask,
            )
            for idx in range(cost_batch.shape[0])
        ],
        dim=0,
    )

    assert batched.shape == cost_batch.shape
    assert torch.allclose(batched, serial, atol=1e-6)


def test_query_partial_transport_uses_episode_axis_and_caps():
    model = _model_with_psr_cfg()
    model.pot_route_cfg.UOT3D_ENTROPIC_EPS = 0.3
    episode_class_ids = torch.tensor([0, 1, 2, 3, 4])
    n_way = int(episode_class_ids.numel())
    feature_dim = n_way
    label_axis_global_labels = episode_class_ids
    target_label_indices = torch.arange(n_way)
    label_axis_text = torch.eye(feature_dim, dtype=torch.float32).index_select(
        0,
        label_axis_global_labels,
    )
    st_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.9, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    point_mask = torch.tensor([[True, True, False], [True, True, True]])
    support_mask = torch.tensor([True, False])
    episode_positive_labels = torch.tensor(
        [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]],
        dtype=torch.float32,
    )
    raw_positive_labels = torch.tensor(
        [[1, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]],
        dtype=torch.float32,
    )

    model.pot_route_cfg.QUERY_PARTIAL_LABEL_CAP = 1e-6
    model.pot_route_cfg.QUERY_PARTIAL_VIS_CAP = 1e-6
    tight_out = model._compute_query_partial_3d_transport(
        st_tokens,
        point_mask,
        label_axis_text,
        label_axis_global_labels,
        target_label_indices,
        support_mask=support_mask,
        episode_positive_labels=episode_positive_labels,
        raw_positive_labels=raw_positive_labels,
    )
    model.pot_route_cfg.QUERY_PARTIAL_LABEL_CAP = 1000.0
    model.pot_route_cfg.QUERY_PARTIAL_VIS_CAP = 1000.0
    loose_out = model._compute_query_partial_3d_transport(
        st_tokens,
        point_mask,
        label_axis_text,
        label_axis_global_labels,
        target_label_indices,
        support_mask=support_mask,
        episode_positive_labels=episode_positive_labels,
        raw_positive_labels=raw_positive_labels,
    )

    assert tight_out["sim"].shape == (n_way, 2, 3)
    assert tight_out["cost"].shape == (n_way, 2, 3)
    assert tight_out["prob_vis"].shape == (n_way, 2, 3)
    assert tight_out["target_vis_prior"].shape == (n_way, 2, 3)
    assert tight_out["vis_prior"].shape == (2, 3)
    assert tight_out["st_transport"].shape == (n_way, 2, 3)
    assert tight_out["shared_transport"].shape == (n_way, 2, 3)
    assert tight_out["transport_mass"].shape == (n_way,)
    assert tight_out["target_label_indices"].detach().cpu().tolist() == list(range(n_way))
    assert tight_out["label_axis_global_labels"].detach().cpu().tolist() == (
        episode_class_ids.tolist()
    )
    valid_cost_mask = point_mask.unsqueeze(0).expand_as(tight_out["cost"])
    assert torch.allclose(
        tight_out["cost"][valid_cost_mask],
        torch.full_like(tight_out["cost"][valid_cost_mask], 0.63),
        atol=1e-6,
    )
    assert tight_out["cost"][:, 0, 2].min() == 1e4
    assert tight_out["st_transport"][:, 0, 2].sum() == 0.0
    assert tight_out["target_vis_prior"][:, 0, 2].sum() == 0.0
    assert torch.all(tight_out["transport_mass"] > 0.0)
    assert torch.all(
        tight_out["st_transport"] <= tight_out["target_vis_prior"] + 1e-5
    )
    assert torch.all(
        tight_out["transport_mass"] <= loose_out["transport_mass"] + 1e-6
    )
    assert not torch.allclose(
        tight_out["st_transport"],
        loose_out["st_transport"],
        atol=1e-6,
    )


def test_query_partial_q2s_uses_episode_candidate_axis_and_ignores_query_labels():
    model = _model_with_psr_cfg()
    model.cfg = SimpleNamespace(POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=False))
    episode_class_ids = torch.tensor([0, 1, 2, 3, 4])
    n_way = int(episode_class_ids.numel())
    num_raw_classes = n_way + 1
    feature_dim = n_way
    model.num_classes = num_raw_classes
    model.atomic_label_names = [str(class_id) for class_id in range(num_raw_classes)]
    model._get_pot_label_text_features = lambda ids, dtype: torch.nn.functional.one_hot(
        ids.cpu(),
        num_classes=feature_dim,
    ).to(dtype=dtype)

    value_tokens = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
                [[0.8, 0.2, 0.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0, 0.0]],
            ],
            [
                [[0.9, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.8, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.1]],
            ],
        ],
        dtype=torch.float32,
    )
    metadata = {
        "support_mask": torch.tensor([True, False, False]),
        "episode_positive_labels": torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0]],
            dtype=torch.float32,
        ),
        "raw_positive_labels": torch.tensor(
            [
                [1, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ],
            dtype=torch.float32,
        ),
        "episode_class_ids": episode_class_ids,
        "pred_visibility": torch.tensor(
            [
                [[True, True], [True, True]],
                [[True, True], [True, False]],
                [[True, True], [True, True]],
            ]
        ),
    }
    route_aux = {
        "support_conditioned_patch_tokens": torch.stack(
            [
                value_tokens.new_full((2, 1, feature_dim), 0.25),
                value_tokens.new_full((2, 1, feature_dim), 0.75),
            ],
            dim=0,
        ),
        "support_branch_class_indices": torch.tensor([0, 2]),
    }

    aux = model._build_query_partial_q2s_aux(
        value_tokens,
        metadata,
        route_aux=route_aux,
    )
    changed_metadata = dict(metadata)
    changed_metadata["raw_positive_labels"] = metadata["raw_positive_labels"].clone()
    changed_metadata["raw_positive_labels"][1:] = torch.tensor(
        [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]],
        dtype=torch.float32,
    )
    changed_aux = model._build_query_partial_q2s_aux(
        value_tokens,
        changed_metadata,
        route_aux=route_aux,
    )

    assert aux["query_partial_label_axis_global_labels"].detach().cpu().tolist() == (
        episode_class_ids.tolist()
    )
    assert aux["query_partial_target_label_indices"].detach().cpu().tolist() == list(
        range(n_way)
    )
    temporal_dim = value_tokens.shape[1]
    assert aux["query_partial_q2s_logits"].shape == (2, n_way)
    # Frame matching keeps the temporal dim: prototypes are [.., T, C].
    assert aux["query_partial_query_prototypes"].shape == (
        2,
        n_way,
        temporal_dim,
        feature_dim,
    )
    assert torch.allclose(
        aux["query_partial_support_prototypes"][0],
        value_tokens.new_full((temporal_dim, feature_dim), 0.25),
    )
    assert torch.allclose(
        aux["query_partial_support_prototypes"][2],
        value_tokens.new_full((temporal_dim, feature_dim), 0.75),
    )
    assert torch.allclose(
        aux["query_partial_q2s_logits"],
        changed_aux["query_partial_q2s_logits"],
        atol=1e-6,
    )
