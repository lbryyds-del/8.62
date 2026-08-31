"""Query-class matchability for multi-label few-shot action recognition.

This extension separates two questions that were coupled by the previous
Query Null token:

1. ``where``: the existing text/support-routed Softmax constructs the
   class-conditioned Query frame prototype;
2. ``whether``: either the legacy Support-calibrated text evidence or the
   positive-versus-confuser Support margin estimates whether the Query-class
   hypothesis is matchable at all.

The matchability is calibrated from labeled Support videos in the current
episode and enters the final q2s logit as a non-positive log-probability
penalty. No Query target is consumed by this module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


_PATCH_MARKER = "_query_class_matchability_original_builder"


def _safe_unit(value: torch.Tensor) -> torch.Tensor:
    """Return finite FP32 unit vectors without creating NaNs for zero rows."""
    value = torch.nan_to_num(
        value.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    """Read a config attribute while also supporting lightweight test stubs."""
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def masked_topk_mean(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    dim: int = -1,
) -> torch.Tensor:
    """Return a masked Top-K mean and zero for fully invalid rows.

    Args:
        scores: Arbitrary floating tensor.
        mask: Boolean tensor broadcastable to ``scores``.
        k: Maximum number of valid values retained along ``dim``.
        dim: Reduction dimension.
    """
    if scores.numel() == 0:
        output_shape = list(scores.shape)
        output_shape.pop(dim % scores.ndim)
        return scores.new_zeros(output_shape, dtype=torch.float32)

    scores_fp32 = torch.nan_to_num(
        scores.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    mask = mask.to(device=scores.device).bool()
    if mask.shape != scores.shape:
        mask = mask.expand_as(scores)

    dim = dim % scores.ndim
    topk = max(1, min(int(k), scores.shape[dim]))
    masked_scores = scores_fp32.masked_fill(~mask, float("-inf"))
    values, indices = torch.topk(masked_scores, k=topk, dim=dim)
    selected_valid = torch.gather(mask, dim, indices)
    values = torch.where(selected_valid, values, torch.zeros_like(values))
    denominator = selected_valid.sum(dim=dim).clamp_min(1).to(values.dtype)
    result = values.sum(dim=dim) / denominator
    any_valid = mask.any(dim=dim)
    result = torch.where(any_valid, result, torch.zeros_like(result))
    return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)


def pairwise_bimhm(
    query_prototypes: torch.Tensor,
    support_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Compute the same bidirectional frame matcher for every pair.

    The normal Pointformer route compares one ``[Q,K,T,D]`` Query tensor with
    one ``[K,T,D]`` positive Support tensor.  The confuser branch needs the
    corresponding score against every negative Support, so this helper
    accepts ``[S,K,T,D]`` and returns ``[Q,K,S]``.  For convenience, the
    single-class form ``[Q,T,D]``/``[S,T,D]`` returns ``[Q,S]``.

    The reductions intentionally mirror ``Pointformer._compute_bidirectional_
    frame_similarity``: a max over the other video's frames followed by a
    mean over the source video's frames in each direction.  Keeping this
    definition identical is important because the positive term is the
    existing q2s score.
    """
    if query_prototypes.ndim not in (3, 4):
        raise ValueError(
            "query_prototypes must have shape [Q,T,D] or [Q,K,T,D]; got "
            f"{tuple(query_prototypes.shape)}."
        )
    if support_prototypes.ndim not in (3, 4):
        raise ValueError(
            "support_prototypes must have shape [S,T,D] or [S,K,T,D]; got "
            f"{tuple(support_prototypes.shape)}."
        )

    squeeze_class = query_prototypes.ndim == 3
    if query_prototypes.ndim == 3:
        query_prototypes = query_prototypes.unsqueeze(1)
    if support_prototypes.ndim == 3:
        support_prototypes = support_prototypes.unsqueeze(1)
    if query_prototypes.shape[1] != support_prototypes.shape[1]:
        raise ValueError(
            "Query and Support class axes must agree; got "
            f"{query_prototypes.shape[1]} and {support_prototypes.shape[1]}."
        )
    if query_prototypes.shape[-1] != support_prototypes.shape[-1]:
        raise ValueError(
            "Query and Support feature dimensions must agree; got "
            f"{query_prototypes.shape[-1]} and {support_prototypes.shape[-1]}."
        )

    query_norm = _safe_unit(query_prototypes)
    support_norm = _safe_unit(support_prototypes)
    # [Q,K,Tq,D] x [S,K,Ts,D] -> [Q,K,Tq,S,Ts].
    sim = torch.einsum("qktd,skud->qktsu", query_norm, support_norm)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
        -1.0,
        1.0,
    )
    # After the Support-time max the dimensions are [Q,K,Tq,S]; average over
    # Query time (dim=2), while retaining the Support-example axis.
    query_to_support = sim.max(dim=-1).values.mean(dim=2)
    support_to_query = sim.max(dim=2).values.mean(dim=-1)
    result = 0.5 * (query_to_support + support_to_query)
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
        -1.0,
        1.0,
    )
    return result[:, 0] if squeeze_class else result


def _aggregate_negative_similarity(
    negative_similarity: torch.Tensor,
    negative_valid: torch.Tensor,
    aggregation: str = "max",
    topk: int = 2,
    temperature: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate ``[Q,K,S]`` confuser scores with a ``[K,S]`` validity mask.

    Returns the aggregate, the index of the hardest individual confuser (for
    diagnostics), and the number of valid confusers per class.  Invalid-only
    classes receive an aggregate of zero and index ``-1``.
    """
    if negative_similarity.ndim != 3:
        raise ValueError(
            "negative_similarity must have shape [Q,K,S]; got "
            f"{tuple(negative_similarity.shape)}."
        )
    if negative_valid.ndim != 2 or tuple(negative_valid.shape) != tuple(
        negative_similarity.shape[1:]
    ):
        raise ValueError(
            "negative_valid must have shape [K,S] matching negative scores; "
            f"got {tuple(negative_valid.shape)}, expected "
            f"{tuple(negative_similarity.shape[1:])}."
        )
    if negative_similarity.shape[-1] == 0:
        shape = negative_similarity.shape[:2]
        zero = negative_similarity.new_zeros(shape, dtype=torch.float32)
        index = torch.full(shape, -1, device=negative_similarity.device, dtype=torch.long)
        count = torch.zeros(
            negative_similarity.shape[1],
            device=negative_similarity.device,
            dtype=torch.long,
        )
        return zero, index, count

    scores = torch.nan_to_num(
        negative_similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    valid = negative_valid.to(device=scores.device).bool()
    valid_broadcast = valid.unsqueeze(0)
    counts = valid.sum(dim=-1)
    masked = scores.masked_fill(~valid_broadcast, float("-inf"))
    hardest, hardest_index = masked.max(dim=-1)
    has_valid = counts.unsqueeze(0) > 0
    hardest = torch.where(has_valid, hardest, torch.zeros_like(hardest))
    hardest_index = torch.where(
        has_valid,
        hardest_index,
        torch.full_like(hardest_index, -1),
    )

    mode = str(aggregation).lower()
    if mode in {"max", "hard_max", "hardest"}:
        aggregate = hardest
    elif mode in {"topk_mean", "topk", "top_k_mean"}:
        k = max(1, min(int(topk), scores.shape[-1]))
        values, indices = torch.topk(masked, k=k, dim=-1)
        selected_valid = torch.gather(valid_broadcast.expand_as(scores), -1, indices)
        values = torch.where(selected_valid, values, torch.zeros_like(values))
        denom = selected_valid.sum(dim=-1).clamp_min(1).to(values.dtype)
        aggregate = values.sum(dim=-1) / denom
        aggregate = torch.where(has_valid, aggregate, torch.zeros_like(aggregate))
    elif mode in {"logmeanexp", "log_mean_exp", "lme"}:
        if temperature <= 0.0:
            raise ValueError("negative aggregation temperature must be positive.")
        # Subtract log(M) so the scale does not depend on how many negatives
        # happen to be present in a multi-label episode.
        lse = torch.logsumexp(masked / float(temperature), dim=-1)
        count_log = counts.clamp_min(1).to(lse.dtype).log().unsqueeze(0)
        aggregate = float(temperature) * (lse - count_log)
        aggregate = torch.where(has_valid, aggregate, torch.zeros_like(aggregate))
    else:
        raise ValueError(
            "NEGATIVE_AGGREGATION must be 'max', 'topk_mean' or "
            f"'logmeanexp'; got {aggregation!r}."
        )

    return (
        torch.nan_to_num(aggregate, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
            -1.0,
            1.0,
        ),
        hardest_index,
        counts,
    )


def _compute_relative_matchability_with_diagnostics(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    negative_valid: torch.Tensor,
    temperature: float = 0.10,
    bias: float = 0.0,
    aggregation: str = "max",
    topk: int = 2,
    negative_temperature: float = 0.10,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Internal relative-matchability computation with diagnostics."""
    if positive_similarity.ndim != 2:
        raise ValueError(
            "positive_similarity must have shape [Q,K]; got "
            f"{tuple(positive_similarity.shape)}."
        )
    if negative_similarity.ndim != 3:
        raise ValueError(
            "negative_similarity must have shape [Q,K,S]; got "
            f"{tuple(negative_similarity.shape)}."
        )
    if tuple(negative_similarity.shape[:2]) != tuple(positive_similarity.shape):
        raise ValueError(
            "Positive and negative similarities must share [Q,K]; got "
            f"{tuple(positive_similarity.shape)} and "
            f"{tuple(negative_similarity.shape)}."
        )
    if temperature <= 0.0:
        raise ValueError("MARGIN_TEMPERATURE must be positive.")

    negative_aggregate, hardest_index, valid_count = _aggregate_negative_similarity(
        negative_similarity,
        negative_valid,
        aggregation=aggregation,
        topk=topk,
        temperature=negative_temperature,
    )
    positive = torch.nan_to_num(
        positive_similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    margin = positive - negative_aggregate
    rho = torch.sigmoid((margin - float(bias)) / float(temperature))
    has_negative = valid_count.unsqueeze(0).to(device=rho.device) > 0
    # No comparison object means the branch is undefined; leave the original
    # q2s score unchanged by assigning neutral rho=1.
    rho = torch.where(has_negative, rho, torch.ones_like(rho))
    margin = torch.where(has_negative, margin, torch.zeros_like(margin))
    negative_aggregate = torch.where(
        has_negative,
        negative_aggregate,
        torch.zeros_like(negative_aggregate),
    )
    hardest_index = torch.where(
        has_negative,
        hardest_index,
        torch.full_like(hardest_index, -1),
    )
    return rho.clamp(0.0, 1.0), margin, negative_aggregate, hardest_index, valid_count


def compute_relative_matchability(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    negative_valid: torch.Tensor,
    temperature: float = 0.10,
    bias: float = 0.0,
    aggregation: str = "max",
    topk: int = 2,
    negative_temperature: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert positive-vs-confuser similarity into ``rho``.

    ``negative_valid`` uses the class-major shape ``[K,S]``.  The public
    return is ``(rho, margin, aggregated_negative_similarity)``; the internal
    builder additionally records the hardest Support index and valid count.
    """
    rho, margin, negative_aggregate, _, _ = (
        _compute_relative_matchability_with_diagnostics(
            positive_similarity,
            negative_similarity,
            negative_valid,
            temperature=temperature,
            bias=bias,
            aggregation=aggregation,
            topk=topk,
            negative_temperature=negative_temperature,
        )
    )
    return rho, margin, negative_aggregate


def build_class_confuser_prototypes(
    pointformer: Any,
    value_tokens: torch.Tensor,
    point_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    query_label_features: torch.Tensor,
    detach_support: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-Support, per-class confuser prototypes.

    A Support is a valid confuser for class ``k`` exactly when its labeled
    episode vector does not contain ``k``.  The route is still class-guided by
    the same Query-side feature ``query_label_features``; no Query labels are
    read.  Returned tensors are ``[S,K,T,D]``, ``[S,K]`` and global Support
    indices ``[S]``.
    """
    if value_tokens.ndim != 4:
        raise ValueError(
            "value_tokens must have shape [B,T,N,D]; got "
            f"{tuple(value_tokens.shape)}."
        )
    batch, temporal_dim, _, feat_dim = value_tokens.shape
    if tuple(point_mask.shape) != tuple(value_tokens.shape[:3]):
        raise ValueError(
            "point_mask must match value_tokens B,T,N; got "
            f"{tuple(point_mask.shape)} and {tuple(value_tokens.shape[:3])}."
        )
    support_mask = support_mask.to(device=value_tokens.device).bool().flatten()
    if tuple(support_mask.shape) != (batch,):
        raise ValueError(
            "support_mask must have shape [B]; got "
            f"{tuple(support_mask.shape)}."
        )
    labels = episode_positive_labels.to(device=value_tokens.device).bool()
    num_classes = query_label_features.shape[0]
    if tuple(labels.shape) != (batch, num_classes):
        raise ValueError(
            "episode_positive_labels must have shape [B,K]; got "
            f"{tuple(labels.shape)}, expected {(batch, num_classes)}."
        )
    if query_label_features.ndim != 2 or query_label_features.shape[-1] != feat_dim:
        raise ValueError(
            "query_label_features must have shape [K,D] matching values; got "
            f"{tuple(query_label_features.shape)} and D={feat_dim}."
        )

    support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
    if support_indices.numel() == 0:
        empty_proto = value_tokens.new_zeros(
            0,
            num_classes,
            temporal_dim,
            feat_dim,
        )
        empty_valid = torch.zeros(
            0,
            num_classes,
            device=value_tokens.device,
            dtype=torch.bool,
        )
        return empty_proto, empty_valid, support_indices

    prototypes = []
    valid_rows = []
    for sample_idx in support_indices.tolist():
        sample_proto, _ = pointformer._compute_frame_softmax_text_prototypes(
            value_tokens[sample_idx],
            point_mask[sample_idx],
            query_label_features,
        )
        if detach_support:
            sample_proto = sample_proto.detach()
        sample_proto = torch.nan_to_num(
            sample_proto,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        frame_valid = sample_proto.float().norm(dim=-1).gt(1e-12).any(dim=-1)
        valid_rows.append((~labels[sample_idx]) & frame_valid)
        prototypes.append(sample_proto)

    return (
        torch.stack(prototypes, dim=0),
        torch.stack(valid_rows, dim=0).bool(),
        support_indices,
    )


def _classwise_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-class means and counts for ``values`` shaped ``[S,K]``."""
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(
            "values and mask must share shape [num_support, num_classes]; got "
            f"{tuple(values.shape)} and {tuple(mask.shape)}."
        )
    weights = mask.to(dtype=values.dtype)
    counts = weights.sum(dim=0)
    means = (values * weights).sum(dim=0) / counts.clamp_min(1.0)
    return means, counts


def compute_matchability_from_similarity(
    similarity: torch.Tensor,
    point_mask: torch.Tensor,
    support_mask: torch.Tensor,
    episode_positive_labels: torch.Tensor,
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    """Compute Support-calibrated Query-class matchability.

    Args:
        similarity: Pure text-to-patch cosine with shape ``[B,K,T,N]``.
        point_mask: Valid trajectory mask with shape ``[B,T,N]``.
        support_mask: Boolean sample split with shape ``[B]``.
        episode_positive_labels: Multi-hot episode labels with shape ``[B,K]``.
            Only Support rows are read; Query rows are deliberately ignored.
        cfg: ``QUERY_CLASS_MATCHABILITY`` config node or a compatible stub.

    Returns:
        Evidence, Support calibration statistics, thresholds and Query
        matchability. Matchability has shape ``[Q,K]``.
    """
    if similarity.ndim != 4:
        raise ValueError(
            "similarity must have shape [B,K,T,N]; got "
            f"{tuple(similarity.shape)}."
        )
    batch, num_classes, temporal_dim, num_points = similarity.shape
    expected_point_shape = (batch, temporal_dim, num_points)
    if tuple(point_mask.shape) != expected_point_shape:
        raise ValueError(
            "point_mask must match similarity B,T,N; got "
            f"{tuple(point_mask.shape)}, expected {expected_point_shape}."
        )
    if tuple(support_mask.shape) != (batch,):
        raise ValueError(
            "support_mask must have shape [B]; got "
            f"{tuple(support_mask.shape)}."
        )
    if tuple(episode_positive_labels.shape) != (batch, num_classes):
        raise ValueError(
            "episode_positive_labels must have shape [B,K]; got "
            f"{tuple(episode_positive_labels.shape)}, expected "
            f"{(batch, num_classes)}."
        )

    point_mask = point_mask.to(device=similarity.device).bool()
    support_mask = support_mask.to(device=similarity.device).bool()
    labels = episode_positive_labels.to(device=similarity.device).bool()
    similarity = torch.nan_to_num(
        similarity.float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)

    patch_topk = int(_cfg_value(cfg, "TOPK_PATCHES", 8))
    frame_topk = int(_cfg_value(cfg, "TOPK_FRAMES", 3))
    if patch_topk <= 0 or frame_topk <= 0:
        raise ValueError("TOPK_PATCHES and TOPK_FRAMES must be positive.")

    expanded_point_mask = point_mask.unsqueeze(1).expand_as(similarity)
    frame_evidence = masked_topk_mean(
        similarity,
        expanded_point_mask,
        patch_topk,
        dim=-1,
    )  # [B,K,T]

    frame_valid = point_mask.any(dim=-1).unsqueeze(1).expand(
        batch,
        num_classes,
        temporal_dim,
    )
    video_evidence = masked_topk_mean(
        frame_evidence,
        frame_valid,
        frame_topk,
        dim=-1,
    )  # [B,K]

    support_evidence = video_evidence[support_mask]
    support_targets = labels[support_mask]
    query_evidence = video_evidence[~support_mask]
    if support_evidence.shape[0] == 0:
        raise ValueError("At least one labeled Support sample is required.")

    positive_mean, positive_count = _classwise_masked_mean(
        support_evidence,
        support_targets,
    )
    negative_mask = ~support_targets
    negative_mean, negative_count = _classwise_masked_mean(
        support_evidence,
        negative_mask,
    )

    all_negative = support_evidence[negative_mask]
    if all_negative.numel() > 0:
        global_negative = all_negative.mean()
    else:
        global_negative = support_evidence.mean()
    negative_mean = torch.where(
        negative_count > 0,
        negative_mean,
        global_negative.expand_as(negative_mean),
    )
    # An episode class should have a positive Support example. The fallback
    # keeps malformed/debug episodes finite without reading Query labels.
    positive_mean = torch.where(
        positive_count > 0,
        positive_mean,
        negative_mean,
    )

    support_gap = positive_mean - negative_mean
    support_reliable = (
        (positive_count > 0)
        & (negative_count > 0)
        & (support_gap > 0.0)
    )
    beta = float(_cfg_value(cfg, "CALIBRATION_BETA", 0.25))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("CALIBRATION_BETA must be in [0, 1].")
    positive_gap = support_gap.clamp_min(0.0)
    threshold = negative_mean + beta * positive_gap

    if bool(_cfg_value(cfg, "DETACH_SUPPORT_STATS", True)):
        positive_mean = positive_mean.detach()
        negative_mean = negative_mean.detach()
        support_gap = support_gap.detach()
        threshold = threshold.detach()

    temperature = float(_cfg_value(cfg, "TEMPERATURE", 0.10))
    if temperature <= 0.0:
        raise ValueError("QUERY_CLASS_MATCHABILITY.TEMPERATURE must be positive.")
    matchability = torch.sigmoid(
        (query_evidence - threshold.unsqueeze(0)) / temperature
    )
    matchability = torch.nan_to_num(
        matchability,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    return {
        "frame_evidence": frame_evidence,
        "video_evidence": video_evidence,
        "query_evidence": query_evidence,
        "support_positive_evidence_mean": positive_mean,
        "support_negative_evidence_mean": negative_mean,
        "support_positive_count": positive_count,
        "support_negative_count": negative_count,
        "support_gap": support_gap,
        "support_reliable": support_reliable,
        "threshold": threshold,
        "matchability": matchability,
    }


def apply_log_matchability_penalty(
    base_logits: torch.Tensor,
    matchability: torch.Tensor,
    cfg: Any,
    support_reliable: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a non-positive log-matchability penalty to q2s logits."""
    if tuple(base_logits.shape) != tuple(matchability.shape):
        raise ValueError(
            "base_logits and matchability must have the same [Q,K] shape; got "
            f"{tuple(base_logits.shape)} and {tuple(matchability.shape)}."
        )
    weight = float(_cfg_value(cfg, "LOG_PENALTY_WEIGHT", 0.25))
    log_eps = float(_cfg_value(cfg, "LOG_EPS", 0.05))
    if weight < 0.0:
        raise ValueError("LOG_PENALTY_WEIGHT must be non-negative.")
    if not 0.0 < log_eps <= 1.0:
        raise ValueError("LOG_EPS must be in (0, 1].")

    penalty = weight * torch.log(matchability.float().clamp_min(log_eps))
    if bool(_cfg_value(cfg, "RELIABILITY_FALLBACK", False)):
        if support_reliable is None:
            raise ValueError(
                "support_reliable is required when RELIABILITY_FALLBACK is enabled."
            )
        support_reliable = support_reliable.to(
            device=penalty.device,
            dtype=torch.bool,
        )
        if tuple(support_reliable.shape) != (penalty.shape[-1],):
            raise ValueError(
                "support_reliable must have shape [K]; got "
                f"{tuple(support_reliable.shape)}, expected {(penalty.shape[-1],)}."
            )
        penalty = torch.where(
            support_reliable.unsqueeze(0),
            penalty,
            torch.zeros_like(penalty),
        )
    final_logits = torch.nan_to_num(
        base_logits.float() + penalty,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    return final_logits, penalty


def _build_frame_softmax_q2s_with_matchability(
    self: Any,
    value_tokens: torch.Tensor,
    metadata: Dict[str, Any],
    pred_tracks: Optional[torch.Tensor] = None,
    matchability_evidence_tokens: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build normal routed prototypes and add Query-class matchability."""
    few_shot_cfg = getattr(getattr(self, "cfg", None), "FEW_SHOT", None)
    cfg = getattr(few_shot_cfg, "QUERY_CLASS_MATCHABILITY", None)
    if not bool(_cfg_value(cfg, "ENABLE", False)):
        original = getattr(self.__class__, _PATCH_MARKER)
        return original(self, value_tokens, metadata, pred_tracks=pred_tracks)
    if bool(getattr(self, "use_query_null_route", False)):
        raise RuntimeError(
            "QUERY_CLASS_MATCHABILITY and QUERY_NULL_ROUTE are mutually "
            "exclusive. Disable the learned Query Null token."
        )

    support_mask = metadata["support_mask"].to(device=value_tokens.device).bool()
    query_mask = ~support_mask
    if not query_mask.any():
        return None

    point_mask = (
        metadata["pred_query_mask"]
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
        else metadata["pred_visibility"]
    ).to(device=value_tokens.device).bool()
    episode_positive_labels = metadata["episode_positive_labels"].to(
        device=value_tokens.device,
    ).bool()
    episode_class_ids = metadata["episode_class_ids"].to(
        device=value_tokens.device,
    ).long()
    episode_class_ids = (
        episode_class_ids[0]
        if episode_class_ids.ndim == 2
        else episode_class_ids
    ).flatten()
    if episode_class_ids.numel() == 0:
        return None

    # Pure text is retained for the independent ``whether`` branch.
    episode_label_text = self._get_pot_label_text_features(
        episode_class_ids,
        value_tokens.dtype,
    )

    evidence_source = str(_cfg_value(cfg, "EVIDENCE_SOURCE", "post")).lower()
    if evidence_source == "post":
        evidence_tokens = value_tokens
    elif evidence_source == "raw":
        if matchability_evidence_tokens is None:
            raise ValueError(
                "Raw matchability evidence was requested, but pre-Pointformer "
                "DinoTxt tokens were not provided."
            )
        evidence_tokens = matchability_evidence_tokens
    else:
        raise ValueError(
            "QUERY_CLASS_MATCHABILITY.EVIDENCE_SOURCE must be 'post' or 'raw'; "
            f"got {evidence_source!r}."
        )
    if tuple(evidence_tokens.shape[:3]) != tuple(value_tokens.shape[:3]):
        raise ValueError(
            "Matchability evidence tokens must align with value token B/T/N; got "
            f"{tuple(evidence_tokens.shape)} and {tuple(value_tokens.shape)}."
        )

    refined_similarity = None
    if bool(getattr(self, "use_cat_cost_aggregation", False)):
        refined_similarity = self._compute_split_cat_refined_point_similarity(
            value_tokens,
            point_mask,
            pred_tracks,
            support_mask,
            episode_positive_labels,
            episode_class_ids,
            episode_label_text,
            raw_positive_labels=metadata.get("raw_positive_labels"),
        )

    support_prototypes = self._build_frame_softmax_support_prototypes(
        value_tokens,
        point_mask,
        support_mask,
        episode_positive_labels,
        episode_label_text,
        precomputed_similarity=refined_similarity,
    )

    # The existing text+Support feature remains responsible for ``where``.
    query_label_features = episode_label_text
    support_visual = None
    support_visual_valid = None
    if bool(getattr(self, "use_support_text_fusion", False)):
        if refined_similarity is not None:
            raise RuntimeError(
                "SUPPORT_TEXT_FUSION cannot consume CAT precomputed query costs."
            )
        (
            query_label_features,
            support_visual,
            support_visual_valid,
        ) = self._fuse_episode_text_with_support_visual(
            episode_label_text,
            support_prototypes,
        )

    query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()
    query_prototypes = []
    for sample_idx in query_indices.tolist():
        if refined_similarity is None:
            sample_prototypes, _ = self._compute_frame_softmax_text_prototypes(
                value_tokens[sample_idx],
                point_mask[sample_idx],
                query_label_features,
            )
        else:
            sample_prototypes, _ = (
                self._compute_frame_softmax_prototypes_from_similarity(
                    value_tokens[sample_idx],
                    point_mask[sample_idx],
                    refined_similarity[sample_idx],
                )
            )
        query_prototypes.append(sample_prototypes.unsqueeze(0))
    if not query_prototypes:
        return None
    query_prototypes = torch.cat(query_prototypes, dim=0)

    diag_similarity = self._compute_bidirectional_frame_similarity(
        query_prototypes,
        support_prototypes,
    )
    alpha = float(getattr(
        self.pot_route_cfg,
        "QUERY_PARTIAL_LOGIT_ALPHA",
        10.0,
    ))
    bias = float(getattr(
        self.pot_route_cfg,
        "QUERY_PARTIAL_LOGIT_BIAS",
        -2.0,
    ))
    base_logits = alpha * diag_similarity + bias

    matchability_mode = str(
        _cfg_value(cfg, "MODE", "threshold")
    ).lower()
    relative_modes = {
        "positive_confuser_margin",
        "positive_confuser",
        "confuser_margin",
    }
    legacy_modes = {"threshold", "legacy", "support_threshold"}
    if (
        matchability_mode not in relative_modes
        and matchability_mode not in legacy_modes
    ):
        raise ValueError(
            "QUERY_CLASS_MATCHABILITY.MODE must be 'threshold' or "
            f"'positive_confuser_margin'; got {matchability_mode!r}."
        )
    if matchability_mode in relative_modes:
        if evidence_source != "post":
            raise ValueError(
                "positive_confuser_margin currently compares Post-Pointformer "
                "prototypes; set EVIDENCE_SOURCE='post'."
            )
        # The current CAT path intentionally computes Support costs only for
        # known positive labels.  Reusing those zero-filled non-positive slots
        # as confusers would make the two sides incomparable, so fail loudly
        # until a CAT-specific negative route is defined.
        if refined_similarity is not None:
            raise RuntimeError(
                "positive_confuser_margin currently requires COST_AGG.ENABLE=False."
            )
        confuser_prototypes, confuser_valid, confuser_support_indices = (
            build_class_confuser_prototypes(
                self,
                value_tokens,
                point_mask,
                support_mask,
                episode_positive_labels,
                query_label_features,
                detach_support=bool(
                    _cfg_value(cfg, "DETACH_CONFUSER_SUPPORT", False)
                ),
            )
        )
        negative_similarity = pairwise_bimhm(
            query_prototypes,
            confuser_prototypes,
        )
        margin_temperature = float(
            _cfg_value(cfg, "MARGIN_TEMPERATURE", 0.10)
        )
        margin_bias = float(_cfg_value(cfg, "MARGIN_BIAS", 0.0))
        negative_aggregation = str(
            _cfg_value(cfg, "NEGATIVE_AGGREGATION", "max")
        )
        negative_topk = int(_cfg_value(cfg, "NEGATIVE_TOPK", 2))
        negative_temperature = float(
            _cfg_value(cfg, "NEGATIVE_TEMPERATURE", margin_temperature)
        )
        (
            matchability,
            relative_margin,
            negative_aggregate,
            hardest_confuser_local,
            confuser_valid_count,
        ) = _compute_relative_matchability_with_diagnostics(
            diag_similarity,
            negative_similarity,
            confuser_valid.transpose(0, 1),
            temperature=margin_temperature,
            bias=margin_bias,
            aggregation=negative_aggregation,
            topk=negative_topk,
            negative_temperature=negative_temperature,
        )

        if confuser_support_indices.numel() > 0:
            safe_local = hardest_confuser_local.clamp_min(0)
            hardest_confuser_global = confuser_support_indices.index_select(
                0,
                safe_local.reshape(-1),
            ).reshape_as(hardest_confuser_local)
            hardest_confuser_global = torch.where(
                hardest_confuser_local.ge(0),
                hardest_confuser_global,
                torch.full_like(hardest_confuser_global, -1),
            )
        else:
            hardest_confuser_global = torch.full(
                hardest_confuser_local.shape,
                -1,
                device=value_tokens.device,
                dtype=torch.long,
            )

        # Keep the generic evidence/threshold names populated so existing
        # consumers can still read a matchability result.  In this mode the
        # evidence is the relative margin and the threshold is the configured
        # margin bias; the richer diagnostics below expose both raw terms.
        matchability_aux = {
            "matchability": matchability,
            "query_evidence": relative_margin,
            "threshold": torch.full(
                (episode_class_ids.numel(),),
                fill_value=margin_bias,
                device=value_tokens.device,
                dtype=torch.float32,
            ),
            "support_reliable": confuser_valid_count > 0,
            "relative_margin": relative_margin,
            "positive_similarity": diag_similarity,
            "negative_similarity": negative_aggregate,
            "hardest_confuser_index": hardest_confuser_global,
            "confuser_valid_count": confuser_valid_count,
            "negative_similarity_all": negative_similarity,
            "confuser_support_indices": confuser_support_indices,
        }
    else:
        pure_text_similarity = self._compute_batched_point_text_similarity(
            evidence_tokens,
            episode_label_text,
        )
        matchability_aux = compute_matchability_from_similarity(
            pure_text_similarity,
            point_mask,
            support_mask,
            episode_positive_labels,
            cfg,
        )
        matchability_aux.update({
            "relative_margin": matchability_aux["query_evidence"],
            "positive_similarity": diag_similarity,
            "negative_similarity": torch.zeros_like(diag_similarity),
            "hardest_confuser_index": torch.full(
                diag_similarity.shape,
                -1,
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "confuser_valid_count": matchability_aux["support_negative_count"],
        })
    apply_during_train = bool(_cfg_value(cfg, "APPLY_DURING_TRAIN", False))
    if bool(getattr(self, "training", False)) and not apply_during_train:
        final_logits = base_logits.float()
        log_penalty = torch.zeros_like(final_logits)
    else:
        final_logits, log_penalty = apply_log_matchability_penalty(
            base_logits,
            matchability_aux["matchability"],
            cfg,
            support_reliable=matchability_aux["support_reliable"],
        )

    target_label_indices = torch.arange(
        episode_class_ids.numel(),
        device=value_tokens.device,
        dtype=torch.long,
    )
    result: Dict[str, torch.Tensor] = {
        "query_partial_q2s_logits": final_logits.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_q2s_base_logits": base_logits.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_query_prototypes": query_prototypes.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_support_prototypes": support_prototypes.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_diag_similarity": diag_similarity.to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_alpha_sim_term": (alpha * diag_similarity).to(
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_bias_term": torch.full_like(
            base_logits,
            fill_value=bias,
            device=value_tokens.device,
            dtype=value_tokens.dtype,
        ),
        "query_partial_query_sample_indices": query_indices.to(
            device=value_tokens.device,
            dtype=torch.long,
        ),
        "query_partial_label_axis_global_labels": episode_class_ids.to(
            device=value_tokens.device,
            dtype=torch.long,
        ),
        "query_partial_target_label_indices": target_label_indices,
        "query_class_matchability": matchability_aux["matchability"],
        "query_class_transport_mass": matchability_aux["matchability"],
        "query_class_evidence": matchability_aux["query_evidence"],
        "query_class_threshold": matchability_aux["threshold"],
        "query_class_log_penalty": log_penalty,
        "query_class_positive_similarity": matchability_aux[
            "positive_similarity"
        ],
        "query_class_hardest_confuser_similarity": matchability_aux[
            "negative_similarity"
        ],
        "query_class_relative_margin": matchability_aux["relative_margin"],
        "query_class_hardest_confuser_support_index": matchability_aux[
            "hardest_confuser_index"
        ],
        "query_class_confuser_valid_count": matchability_aux[
            "confuser_valid_count"
        ],
    }
    if "support_positive_evidence_mean" in matchability_aux:
        result.update({
            "support_positive_evidence_mean": matchability_aux[
                "support_positive_evidence_mean"
            ],
            "support_negative_evidence_mean": matchability_aux[
                "support_negative_evidence_mean"
            ],
            "support_positive_evidence_count": matchability_aux[
                "support_positive_count"
            ],
            "support_negative_evidence_count": matchability_aux[
                "support_negative_count"
            ],
            "support_evidence_gap": matchability_aux["support_gap"],
            "support_calibration_reliable": matchability_aux[
                "support_reliable"
            ],
        })
    else:
        result["query_class_confuser_available"] = matchability_aux[
            "support_reliable"
        ]
    if "negative_similarity_all" in matchability_aux:
        # This tensor is useful for offline diagnostics, but is intentionally
        # not consumed by the q2s loss.  Keep it out of CSV rows to avoid
        # materializing one column per Support.
        result["query_class_confuser_similarities"] = matchability_aux[
            "negative_similarity_all"
        ]
        result["query_class_confuser_support_indices"] = matchability_aux[
            "confuser_support_indices"
        ]
    if support_visual is not None:
        result.update({
            "support_text_fusion_query_features": query_label_features.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "support_text_fusion_visual_prototypes": support_visual.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "support_text_fusion_valid_classes": support_visual_valid.to(
                device=value_tokens.device,
                dtype=torch.bool,
            ),
        })
    return result


def install_query_class_matchability(pointformer_cls: Any) -> None:
    """Install the extension once without editing the large Pointformer file."""
    if hasattr(pointformer_cls, _PATCH_MARKER):
        return
    setattr(
        pointformer_cls,
        _PATCH_MARKER,
        pointformer_cls._build_frame_softmax_q2s_aux,
    )
    pointformer_cls._build_frame_softmax_q2s_aux = (
        _build_frame_softmax_q2s_with_matchability
    )
