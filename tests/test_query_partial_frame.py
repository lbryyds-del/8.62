"""Unit tests for the B' query-partial frame-matching path (QUERY_PARTIAL_FRAME_MATCH).

Mirrors tests/test_psr_3d_uot.py: exercises the new helpers in isolation via
Pointformer.__new__ + a minimal pot_route_cfg, so no weights/data are needed.
"""

from types import SimpleNamespace

import torch

from trokens.models.pointformer import Pointformer


def _stub_model():
    model = Pointformer.__new__(Pointformer)
    model.pot_route_cfg = SimpleNamespace(
        QUERY_PARTIAL_LOGIT_ALPHA=10.0,
        QUERY_PARTIAL_LOGIT_BETA=1.0,
        QUERY_PARTIAL_LOGIT_BIAS=-2.0,
    )
    return model


def test_frame_logits_shape_and_range():
    model = _stub_model()
    num_q, num_class, temporal, dim = 3, 5, 8, 16
    query_frame = torch.randn(num_q, num_class, temporal, dim)
    support_frame = torch.randn(num_class, temporal, dim)
    mass = torch.rand(num_q, num_class)

    logits, bidir = model._compute_query_partial_frame_logits(
        query_frame, support_frame, mass
    )

    assert logits.shape == (num_q, num_class)
    assert bidir.shape == (num_q, num_class)
    assert torch.isfinite(logits).all()
    assert (bidir <= 1.0 + 1e-4).all() and (bidir >= -1.0 - 1e-4).all()


def test_frame_logits_diagonal_self_match_is_one():
    # When the query class-c prototype equals the support class-c prototype, the
    # bidirectional nearest-neighbor cosine on the diagonal should be ~1.
    model = _stub_model()
    num_class, temporal, dim = 4, 6, 8
    support_frame = torch.randn(num_class, temporal, dim)
    query_frame = support_frame.unsqueeze(0).clone()  # [1, N, T, C]
    mass = torch.zeros(1, num_class)

    logits, bidir = model._compute_query_partial_frame_logits(
        query_frame, support_frame, mass
    )

    assert torch.allclose(bidir[0], torch.ones(num_class), atol=1e-4)
    # Identical prototypes => bidir cosine ~1 => logit = alpha*1 + beta*0 + bias.
    expected = 10.0 * 1.0 + 1.0 * 0.0 + (-2.0)
    assert torch.allclose(logits[0], torch.full((num_class,), expected), atol=1e-3)


def test_support_prototypes_frame_keeps_time_and_handles_missing_class():
    model = _stub_model()
    num_support, temporal, points, dim = 4, 8, 10, 16
    value_tokens = torch.randn(num_support, temporal, points, dim)
    point_mask = torch.ones(num_support, temporal, points, dtype=torch.bool)
    support_mask = torch.tensor([True, True, False, False])
    num_class = 5
    episode_positive_labels = torch.zeros(num_support, num_class)
    episode_positive_labels[0, 1] = 1.0  # support sample 0 -> class 1
    episode_positive_labels[1, 3] = 1.0  # support sample 1 -> class 3

    proto = model._build_query_partial_support_prototypes_frame(
        value_tokens, point_mask, support_mask, episode_positive_labels, route_aux=None
    )

    assert proto.shape == (num_class, temporal, dim)   # time dimension preserved
    assert torch.isfinite(proto).all()
    assert proto[1].abs().sum() > 0                    # class 1 has a support sample
    assert proto[3].abs().sum() > 0                    # class 3 has a support sample
    assert proto[0].abs().sum() == 0                   # class 0 has none -> zero proto
