# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Copyright (c) 2023 KU-CVLAB
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Original implementation by Seokju Cho and Heeseong Shin:
# https://github.com/cvlab-kaist/CAT-Seg
#
# This file contains a project-local adaptation of CAT-Seg's spatial
# aggregation blocks.  It intentionally omits CAT-Seg's class transformer
# and decoder so frames and classes remain independent batch branches.
# --------------------------------------------------------

"""Frame- and class-independent CAT-Seg spatial cost aggregation."""

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trokens.models.common import DropPath, Mlp


Resolution = Union[int, Sequence[int]]


def _to_2tuple(value: Resolution, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
        if len(pair) != 2:
            raise ValueError(f"{name} must be an int or a length-2 sequence")
    if not all(isinstance(item, int) and item > 0 for item in pair):
        raise ValueError(f"{name} values must be positive integers, got {pair}")
    return pair


def _window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """Partition ``[B, H, W, C]`` into flattened non-overlapping windows."""

    batch, height, width, channels = x.shape
    window_h, window_w = window_size
    x = x.reshape(
        batch,
        height // window_h,
        window_h,
        width // window_w,
        window_w,
        channels,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(-1, window_h * window_w, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    resolution: Tuple[int, int],
) -> torch.Tensor:
    """Reverse :func:`_window_partition` into ``[B, H, W, C]``."""

    height, width = resolution
    window_h, window_w = window_size
    windows_per_branch = (height // window_h) * (width // window_w)
    batch = windows.shape[0] // windows_per_branch
    channels = windows.shape[-1]
    x = windows.reshape(
        batch,
        height // window_h,
        width // window_w,
        window_h,
        window_w,
        channels,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch, height, width, channels)
    )


class _GuidedWindowAttention(nn.Module):
    """CAT-Seg window attention with appearance guidance in Q/K only."""

    def __init__(
        self,
        cost_dim: int,
        guidance_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.guidance_dim = guidance_dim
        self.num_heads = num_heads
        self.head_dim = cost_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Appearance guidance controls affinities, while values always contain
        # cost features only.  This is the key CAT-Seg spatial-block semantic.
        qk_dim = cost_dim + guidance_dim
        self.q = nn.Linear(qk_dim, cost_dim, bias=qkv_bias)
        self.k = nn.Linear(qk_dim, cost_dim, bias=qkv_bias)
        self.v = nn.Linear(cost_dim, cost_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(cost_dim, cost_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        cost_and_guidance: torch.Tensor,
        attention_mask: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_windows, token_count, _ = cost_and_guidance.shape
        cost = cost_and_guidance[..., : self.cost_dim]

        q = self.q(cost_and_guidance).reshape(
            batch_windows, token_count, self.num_heads, self.head_dim
        )
        k = self.k(cost_and_guidance).reshape(
            batch_windows, token_count, self.num_heads, self.head_dim
        )
        v = self.v(cost).reshape(
            batch_windows, token_count, self.num_heads, self.head_dim
        )
        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attention = q @ k.transpose(-2, -1)
        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            attention = attention.reshape(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                token_count,
                token_count,
            )
            attention = attention + attention_mask[None, :, None, :, :]
            attention = attention.reshape(
                batch_windows, self.num_heads, token_count, token_count
            )
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=attention.device).bool()
            if tuple(valid_mask.shape) != (batch_windows, token_count):
                raise ValueError(
                    "valid_mask must match the window/token dimensions; got "
                    f"{tuple(valid_mask.shape)}, expected "
                    f"{(batch_windows, token_count)}"
                )
            # Empty raster cells cannot contribute keys or values.  Finite
            # masking avoids NaNs for a window containing no occupied cells;
            # such a window has no valid queries and is zeroed below.
            attention = attention.masked_fill(
                ~valid_mask[:, None, None, :],
                -100.0,
            )
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        output = (attention @ v).transpose(1, 2).reshape(
            batch_windows, token_count, self.cost_dim
        )
        output = self.proj(output)
        output = self.proj_dropout(output)
        if valid_mask is not None:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        return output


class _GuidedSwinBlock(nn.Module):
    """One CAT-Seg W-MSA or SW-MSA cost block."""

    def __init__(
        self,
        cost_dim: int,
        guidance_dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        proj_dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.cost_norm = nn.LayerNorm(cost_dim)
        self.attention = _GuidedWindowAttention(
            cost_dim=cost_dim,
            guidance_dim=guidance_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(cost_dim)
        self.ffn = Mlp(
            in_features=cost_dim,
            hidden_features=int(cost_dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=proj_dropout,
        )
        self.register_buffer(
            "attention_mask",
            self._make_shifted_window_mask(),
            persistent=False,
        )

    def _make_shifted_window_mask(self) -> torch.Tensor:
        shift_h, shift_w = self.shift_size
        if shift_h == 0 and shift_w == 0:
            return None

        height, width = self.input_resolution
        window_h, window_w = self.window_size
        image_mask = torch.zeros((1, height, width, 1))

        height_slices = (
            (
                slice(0, -window_h),
                slice(-window_h, -shift_h),
                slice(-shift_h, None),
            )
            if shift_h
            else (slice(0, None),)
        )
        width_slices = (
            (
                slice(0, -window_w),
                slice(-window_w, -shift_w),
                slice(-shift_w, None),
            )
            if shift_w
            else (slice(0, None),)
        )
        region_id = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice, :] = region_id
                region_id += 1

        mask_windows = _window_partition(image_mask, self.window_size).squeeze(-1)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attention_mask.masked_fill(
            attention_mask != 0, -100.0
        ).masked_fill(attention_mask == 0, 0.0)

    def forward(
        self,
        cost_tokens: torch.Tensor,
        guidance_tokens: torch.Tensor,
        spatial_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        height, width = self.input_resolution
        batch, token_count, channels = cost_tokens.shape
        if token_count != height * width or channels != self.cost_dim:
            raise ValueError(
                "cost token shape does not match the configured resolution/dimension: "
                f"got {tuple(cost_tokens.shape)}, expected [B, {height * width}, "
                f"{self.cost_dim}]"
            )

        shortcut = cost_tokens
        cost = self.cost_norm(cost_tokens).reshape(batch, height, width, channels)
        guidance = guidance_tokens.reshape(batch, height, width, -1)
        cost_and_guidance = torch.cat((cost, guidance), dim=-1)
        mask = None
        if spatial_mask is not None:
            if tuple(spatial_mask.shape) != (batch, token_count):
                raise ValueError(
                    "spatial_mask must match the cost token dimensions; got "
                    f"{tuple(spatial_mask.shape)}, expected {(batch, token_count)}"
                )
            mask = spatial_mask.to(device=cost_tokens.device).bool().reshape(
                batch, height, width, 1
            )
            cost_and_guidance = cost_and_guidance * mask.to(
                cost_and_guidance.dtype
            )

        shift_h, shift_w = self.shift_size
        if shift_h or shift_w:
            cost_and_guidance = torch.roll(
                cost_and_guidance,
                shifts=(-shift_h, -shift_w),
                dims=(1, 2),
            )
            if mask is not None:
                mask = torch.roll(
                    mask,
                    shifts=(-shift_h, -shift_w),
                    dims=(1, 2),
                )

        windows = _window_partition(cost_and_guidance, self.window_size)
        window_mask = None
        if mask is not None:
            window_mask = _window_partition(mask, self.window_size).squeeze(-1)
        attended_windows = self.attention(
            windows,
            self.attention_mask,
            valid_mask=window_mask,
        )
        attended = _window_reverse(
            attended_windows,
            self.window_size,
            self.input_resolution,
        )

        if shift_h or shift_w:
            attended = torch.roll(
                attended,
                shifts=(shift_h, shift_w),
                dims=(1, 2),
            )
        attended = attended.reshape(batch, height * width, channels)

        cost_tokens = shortcut + self.drop_path(attended)
        cost_tokens = cost_tokens + self.drop_path(
            self.ffn(self.ffn_norm(cost_tokens))
        )
        if spatial_mask is not None:
            cost_tokens = cost_tokens * spatial_mask.unsqueeze(-1).to(
                cost_tokens.dtype
            )
        return cost_tokens


class _SpatialAggregationPair(nn.Module):
    """A CAT-Seg W-MSA -> SW-MSA pair with shared input guidance."""

    def __init__(
        self,
        cost_dim: int,
        guidance_dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: Tuple[int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        proj_dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.guidance_norm = nn.LayerNorm(guidance_dim)
        no_shift = (0, 0)
        shifted = (window_size[0] // 2, window_size[1] // 2)
        self.window_attention = _GuidedSwinBlock(
            cost_dim,
            guidance_dim,
            input_resolution,
            num_heads,
            window_size,
            no_shift,
            mlp_ratio,
            qkv_bias,
            proj_dropout,
            attn_dropout,
            drop_path,
        )
        self.shifted_window_attention = _GuidedSwinBlock(
            cost_dim,
            guidance_dim,
            input_resolution,
            num_heads,
            window_size,
            shifted,
            mlp_ratio,
            qkv_bias,
            proj_dropout,
            attn_dropout,
            drop_path,
        )

    def forward(
        self,
        cost_tokens: torch.Tensor,
        guidance_tokens: torch.Tensor,
        spatial_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        guidance_tokens = self.guidance_norm(guidance_tokens)
        if spatial_mask is not None:
            guidance_tokens = guidance_tokens * spatial_mask.unsqueeze(-1).to(
                guidance_tokens.dtype
            )
        cost_tokens = self.window_attention(
            cost_tokens,
            guidance_tokens,
            spatial_mask=spatial_mask,
        )
        return self.shifted_window_attention(
            cost_tokens,
            guidance_tokens,
            spatial_mask=spatial_mask,
        )


class CATSpatialCostAggregator(nn.Module):
    """Refine dense text/patch costs independently per frame and class.

    Args:
        appearance_dim: Channel dimension ``D`` of the dense patch features.
        input_resolution: Fixed spatial resolution ``(H, W)``.
        cost_dim: Hidden cost embedding dimension.
        guidance_dim: Projected appearance-guidance dimension.
        num_heads: Attention heads. ``cost_dim`` must be divisible by it.
        window_size: Fixed window shape. Both spatial dimensions must be
            divisible by it.
        num_layers: Number of W-MSA -> SW-MSA pairs.

    Forward inputs:
        dense_patch: ``[B, T, H, W, D]`` dense, ordered patch grid.
        text_features: ``[K, D]`` shared episode class text features.

    ``forward_precomputed`` additionally accepts a precomputed correlation,
    dense guidance, and an occupancy mask.  This is used when irregular
    post-Pointformer trajectory tokens are scattered back onto their current
    frame coordinates.  Empty raster cells stay zero and never participate in
    window attention.

    Returns:
        A refined cost of shape ``[B, T, K, H, W]``. The caller can sample
        this dense output at trajectory coordinates and use it directly in
        place of the original point/text cosine.

    ``B * T * K`` is folded into the batch axis before every convolution and
    attention operation, so branches share parameters but never exchange
    content.  There is deliberately no temporal or class attention here.
    """

    def __init__(
        self,
        appearance_dim: int,
        input_resolution: Resolution = (16, 16),
        cost_dim: int = 32,
        guidance_dim: int = 32,
        num_heads: int = 4,
        window_size: Resolution = 4,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_resolution = _to_2tuple(input_resolution, "input_resolution")
        self.window_size = _to_2tuple(window_size, "window_size")

        if not isinstance(appearance_dim, int) or appearance_dim <= 0:
            raise ValueError("appearance_dim must be a positive integer")
        if not isinstance(cost_dim, int) or cost_dim <= 0:
            raise ValueError("cost_dim must be a positive integer")
        if not isinstance(guidance_dim, int) or guidance_dim <= 0:
            raise ValueError("guidance_dim must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if cost_dim % num_heads != 0:
            raise ValueError(
                f"cost_dim ({cost_dim}) must be divisible by num_heads ({num_heads})"
            )
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        for name, value in (
            ("proj_dropout", proj_dropout),
            ("attn_dropout", attn_dropout),
            ("drop_path", drop_path),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        height, width = self.input_resolution
        window_h, window_w = self.window_size
        if window_h > height or window_w > width:
            raise ValueError(
                f"window_size {self.window_size} cannot exceed input_resolution "
                f"{self.input_resolution}"
            )
        if height % window_h != 0 or width % window_w != 0:
            raise ValueError(
                f"input_resolution {self.input_resolution} must be divisible by "
                f"window_size {self.window_size}"
            )

        self.appearance_dim = appearance_dim
        self.cost_dim = cost_dim
        self.guidance_dim = guidance_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # CAT-Seg's cost embedding and appearance-guidance projection.
        self.cost_embedding = nn.Conv2d(
            1, cost_dim, kernel_size=7, stride=1, padding=3
        )
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(
                appearance_dim,
                guidance_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.spatial_layers = nn.ModuleList(
            [
                _SpatialAggregationPair(
                    cost_dim=cost_dim,
                    guidance_dim=guidance_dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_dropout=proj_dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                )
                for _ in range(num_layers)
            ]
        )
        self.cost_head = nn.Conv2d(
            cost_dim, 1, kernel_size=3, stride=1, padding=1
        )

    @staticmethod
    def compute_correlation(
        dense_patch: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FP32 cosine costs as ``[B, T, K, H, W]``."""

        if dense_patch.ndim != 5:
            raise ValueError(
                "dense_patch must have shape [B, T, H, W, D], got "
                f"{tuple(dense_patch.shape)}"
            )
        if text_features.ndim != 2:
            raise ValueError(
                "text_features must have shape [K, D], got "
                f"{tuple(text_features.shape)}"
            )
        if dense_patch.shape[-1] != text_features.shape[-1]:
            raise ValueError(
                "dense_patch and text_features channel dimensions differ: "
                f"{dense_patch.shape[-1]} versus {text_features.shape[-1]}"
            )
        if dense_patch.shape[0] == 0 or dense_patch.shape[1] == 0:
            raise ValueError("dense_patch batch and frame dimensions must be non-empty")
        if text_features.shape[0] == 0:
            raise ValueError("text_features must contain at least one class")

        # Explicitly disable the surrounding AMP context for the cosine.  The
        # cast remains differentiable for fp16/bfloat16 inputs.
        with torch.autocast(device_type=dense_patch.device.type, enabled=False):
            patch_norm = F.normalize(dense_patch.float(), dim=-1)
            text_norm = F.normalize(text_features.float(), dim=-1)
            return torch.einsum("bthwd,kd->btkhw", patch_norm, text_norm)

    def forward(
        self,
        dense_patch: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        if dense_patch.ndim != 5:
            raise ValueError(
                "dense_patch must have shape [B, T, H, W, D], got "
                f"{tuple(dense_patch.shape)}"
            )
        batch, frames, height, width, channels = dense_patch.shape
        if (height, width) != self.input_resolution:
            raise ValueError(
                f"dense_patch resolution {(height, width)} does not match the "
                f"configured input_resolution {self.input_resolution}"
            )
        if channels != self.appearance_dim:
            raise ValueError(
                f"dense_patch channel dimension {channels} does not match "
                f"appearance_dim {self.appearance_dim}"
            )

        correlation = self.compute_correlation(dense_patch, text_features)
        occupancy_mask = torch.ones(
            batch,
            frames,
            height,
            width,
            device=dense_patch.device,
            dtype=torch.bool,
        )
        return self.forward_precomputed(
            correlation,
            dense_patch,
            occupancy_mask,
        )

    def forward_precomputed(
        self,
        correlation: torch.Tensor,
        dense_guidance: torch.Tensor,
        occupancy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate a masked dense cost reconstructed from trajectory tokens.

        Args:
            correlation: Point/text cosine rasterized as ``[B,T,K,H,W]``.
            dense_guidance: Post-Pointformer point features rasterized as
                ``[B,T,H,W,D]``.
            occupancy_mask: ``[B,T,H,W]`` boolean mask.  False cells are true
                holes rather than valid zero-valued costs.
        """
        if correlation.ndim != 5:
            raise ValueError(
                "correlation must have shape [B,T,K,H,W], got "
                f"{tuple(correlation.shape)}"
            )
        if dense_guidance.ndim != 5:
            raise ValueError(
                "dense_guidance must have shape [B,T,H,W,D], got "
                f"{tuple(dense_guidance.shape)}"
            )
        if occupancy_mask.ndim != 4:
            raise ValueError(
                "occupancy_mask must have shape [B,T,H,W], got "
                f"{tuple(occupancy_mask.shape)}"
            )

        batch, frames, classes, height, width = correlation.shape
        expected_guidance = (batch, frames, height, width, self.appearance_dim)
        if tuple(dense_guidance.shape) != expected_guidance:
            raise ValueError(
                "dense_guidance shape does not match correlation/resolution: "
                f"got {tuple(dense_guidance.shape)}, expected "
                f"{expected_guidance}"
            )
        expected_mask = (batch, frames, height, width)
        if tuple(occupancy_mask.shape) != expected_mask:
            raise ValueError(
                "occupancy_mask shape does not match correlation: got "
                f"{tuple(occupancy_mask.shape)}, expected {expected_mask}"
            )
        if (height, width) != self.input_resolution:
            raise ValueError(
                f"correlation resolution {(height, width)} does not match the "
                f"configured input_resolution {self.input_resolution}"
            )
        if classes == 0:
            raise ValueError("correlation must contain at least one class")

        correlation = torch.nan_to_num(
            correlation.float(),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        dense_guidance = torch.nan_to_num(
            dense_guidance.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        occupancy_mask = occupancy_mask.to(
            device=correlation.device,
            dtype=torch.bool,
        )
        classes = correlation.shape[2]
        branches = batch * frames * classes

        # Cost embedding is applied to every (video, frame, class) branch.
        branch_mask = (
            occupancy_mask.unsqueeze(2)
            .expand(-1, -1, classes, -1, -1)
            .reshape(branches, 1, height, width)
        )
        cost = correlation.reshape(branches, 1, height, width)
        cost = cost * branch_mask.to(cost.dtype)
        cost = self.cost_embedding(cost)
        cost = cost * branch_mask.to(cost.dtype)
        cost_tokens = cost.permute(0, 2, 3, 1).reshape(
            branches, height * width, self.cost_dim
        )

        # Guidance is frame-specific, then repeated (without interaction)
        # across the class branches belonging to that frame.
        frame_mask = occupancy_mask.reshape(batch * frames, 1, height, width)
        appearance = dense_guidance.permute(0, 1, 4, 2, 3).reshape(
            batch * frames, self.appearance_dim, height, width
        )
        appearance = appearance * frame_mask.to(appearance.dtype)
        guidance = self.guidance_projection(appearance)
        guidance = guidance * frame_mask.to(guidance.dtype)
        guidance = guidance.permute(0, 2, 3, 1).reshape(
            batch, frames, height * width, self.guidance_dim
        )
        guidance = (
            guidance.unsqueeze(2)
            .expand(-1, -1, classes, -1, -1)
            .reshape(branches, height * width, self.guidance_dim)
        )
        token_mask = branch_mask.reshape(branches, height * width)

        for spatial_layer in self.spatial_layers:
            cost_tokens = spatial_layer(
                cost_tokens,
                guidance,
                spatial_mask=token_mask,
            )

        cost = cost_tokens.reshape(
            branches, height, width, self.cost_dim
        ).permute(0, 3, 1, 2)
        refined_cost = self.cost_head(cost)
        refined_cost = refined_cost * branch_mask.to(refined_cost.dtype)
        return refined_cost.reshape(batch, frames, classes, height, width)


__all__ = ["CATSpatialCostAggregator"]
