"""Query-class matchability for multi-label few-shot action recognition.

This extension separates two questions that were coupled by the previous
Query Null token:

1. ``where``: the existing text/support-routed Softmax constructs the
   class-conditioned Query frame prototype;
2. ``whether``: pure text-to-visual evidence estimates whether the Query-class
   hypothesis is matchable at all.

The matchability is calibrated from labeled Support videos in the current
episode and enters the final q2s logit as a non-positive log-probability
penalty. No Query target is consumed by this module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


_PATCH_MARKER = "_query_class_matchability_original_builder"


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

    beta = float(_cfg_value(cfg, "CALIBRATION_BETA", 0.25))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("CALIBRATION_BETA must be in [0, 1].")
    positive_gap = (positive_mean - negative_mean).clamp_min(0.0)
    threshold = negative_mean + beta * positive_gap

    if bool(_cfg_value(cfg, "DETACH_SUPPORT_STATS", True)):
        positive_mean = positive_mean.detach()
        negative_mean = negative_mean.detach()
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
        "threshold": threshold,
        "matchability": matchability,
    }


def apply_log_matchability_penalty(
    base_logits: torch.Tensor,
    matchability: torch.Tensor,
    cfg: Any,
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
) -> Optional[Dict[str, torch.Tensor]]:
    """Build normal routed prototypes and add Query-class matchability."""
    cfg = getattr(self.cfg.FEW_SHOT, "QUERY_CLASS_MATCHABILITY", None)
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

    pure_text_similarity = self._compute_batched_point_text_similarity(
        value_tokens,
        episode_label_text,
    )
    matchability_aux = compute_matchability_from_similarity(
        pure_text_similarity,
        point_mask,
        support_mask,
        episode_positive_labels,
        cfg,
    )
    final_logits, log_penalty = apply_log_matchability_penalty(
        base_logits,
        matchability_aux["matchability"],
        cfg,
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
    }
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
