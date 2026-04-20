"""Pointformer model."""
import os
import sys
import json
from functools import partial
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sympy import divisors
import numpy as np
from torch.nn.init import trunc_normal_

from trokens.models.attention import TrajectoryAttentionBlock
from trokens.models.branches.motion_blocks import (
    CrossMotionModule,
    HODMotionModule
)
from trokens.datasets.hod import get_orientation_hist
from .build import MODEL_REGISTRY

# pylint: disable=unused-argument,redefined-builtin

@MODEL_REGISTRY.register()
class Pointformer(nn.Module):
    """ Main model for point tracking based transformer model.
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        # self.patch_size = cfg.MF.PATCH_SIZE
        self.feat_extractor_type = cfg.MODEL.FEAT_EXTRACTOR
        if self.feat_extractor_type == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            vit_mode = dino_config.split("_")[1]
            if 'vits' in vit_mode:
                vit_type = 'vits'
                self.embed_dim = self.dino_feat_size = 384
            elif 'vitb' in vit_mode:
                vit_type = 'vitb'
                self.embed_dim = self.dino_feat_size = 768
            elif 'vitl' in vit_mode:
                vit_type = 'vitl'
                self.embed_dim = self.dino_feat_size = 1024

            else:
                raise NotImplementedError("Only supports ViT-B and ViT-S for DINO")
            self.patch_size = int(vit_mode.replace(vit_type, ""))
        elif self.feat_extractor_type == "clip_vit_b16":
            self.embed_dim = 768
            self.patch_size = 16
        elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
            self.embed_dim = 1024
            self.patch_size = 14
        else:
            raise NotImplementedError('Feature extractor not implemented')


        self.in_chans = cfg.MF.CHANNELS
        if cfg.TRAIN.DATASET == "epickitchens" and cfg.TASK == 'classification':
            self.num_classes = [97, 300]
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        self.depth = cfg.MF.DEPTH
        self.num_heads = cfg.MF.NUM_HEADS
        self.mlp_ratio = cfg.MF.MLP_RATIO
        self.qkv_bias = cfg.MF.QKV_BIAS
        self.drop_rate = cfg.MF.DROP
        self.drop_path_rate = cfg.MF.DROP_PATH
        self.head_dropout = cfg.MF.HEAD_DROPOUT
        self.video_input = cfg.MF.VIDEO_INPUT
        self.temporal_resolution = cfg.DATA.NUM_FRAMES
        self.use_mlp = cfg.MF.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MF.ATTN_DROPOUT
        self.head_act = cfg.MF.HEAD_ACT
        self.cfg = cfg
        self.text_cluster_cfg = cfg.FEW_SHOT.TEXT_CLUSTER
        self.use_text_conditioned_support = (
            cfg.TASK == 'few_shot'
            and cfg.DATA.MULTI_LABEL
            and self.feat_extractor_type == "clip_vit_b16"
            and self.text_cluster_cfg.ENABLE
            and not cfg.MODEL.APPEARANCE_MODULE_DISABLE
        )
        self.num_patches = (224 // self.patch_size) ** 2
        if cfg.POINT_INFO.ENABLE:
            self.point_grid_size = self.get_point_grid_size()

        else:
            self.point_grid_size = int(self.num_patches ** 0.5)

        # CLS token
        if cfg.MODEL.USE_CLS_TOKEN:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = nn.Identity()

        # # Positional embedding

        self.pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)

        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        ##
        blocks = []
        for i in range(self.depth):
            # pt_attention is introduced, for now its just space-time attention
            _block = TrajectoryAttentionBlock(
                cfg = cfg,
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pt_attention=cfg.MF.PT_ATTENTION,
                use_pt_visibility=cfg.MF.USE_PT_VISIBILITY or cfg.POINT_INFO.USE_PT_QUERY_MASK,
                num_mlp_layers=cfg.MF.NUM_MLP_LAYERS,
            )

            blocks.append(_block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(self.embed_dim)
        if self.cfg.MODEL.APPEARANCE_MODULE_DISABLE:
            assert (
            self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE or
            self.cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE), "One motion module must be enabled"

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh()
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
            self.agg_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.agg_cls_token, std=.02)


        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, f'head{a}', nn.Linear(
                                        self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes)
                if self.num_classes > 0 else nn.Identity())
        self.patch_num_side = 224 // self.patch_size
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.patch_num_side,
                                                        self.patch_num_side))

        trunc_normal_(self.spatial_pos_embed, std=.02)
        if cfg.MODEL.FEAT_EXTRACTOR == 'resnet':
            #TODO(pulkit): Remove hard coding
            self.space_pos_embed = nn.Parameter(torch.zeros(1,49, self.embed_dim))
        else:
            self.space_pos_embed = nn.Parameter(
                                torch.zeros(1,self.num_patches, self.embed_dim))

        self.time_pos_embed = nn.Parameter(
                        torch.zeros(1,self.cfg.DATA.NUM_FRAMES, self.embed_dim))
        trunc_normal_(self.space_pos_embed, std=.02)
        trunc_normal_(self.time_pos_embed, std=.02)
        self.space_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.time_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)



        self.spatial_pos_embed_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.layer_to_use = None

        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)
        if self.feat_extractor_type == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            torch_home = os.environ.get("TORCH_HOME", os.path.join(os.getcwd(), ".torch-cache"))
            os.environ.setdefault("TORCH_HOME", torch_home)
            local_path = os.path.join(torch_home, 'hub')
            if 'v2' in dino_config:
                local_path = os.path.join(local_path , 'facebookresearch_dinov2_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
            else:
                local_path = os.path.join(local_path , 'facebookresearch_dino_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
                self.dino.num_register_tokens = 0

            self.feat_dict = dict()
            #output of last norm to be taken.
            layer = self.dino.norm
            self.hook = layer.register_forward_hook(self.hook_fn(self.feat_dict, 'dino'))

            self.dino.cuda()
            # Set all DINO parameters to not require gradients
            for param in self.dino.parameters():
                param.requires_grad = False
        elif self.feat_extractor_type == "clip_vit_b16":
            import clip

            clip_model, _ = clip.load("ViT-B/16", device="cuda", jit=False)
            self.clip_model = clip_model
            self.clip_tokenize = clip.tokenize
            self.clip_visual = clip_model.visual
            self.clip_model.cuda()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            if self.use_text_conditioned_support:
                prompt_bank_path = self._resolve_prompt_bank_path(
                    self.text_cluster_cfg.PROMPT_BANK_PATH
                )
                prompt_embeddings = self._load_prompt_bank_embeddings(prompt_bank_path)
                self.register_buffer(
                    "text_cluster_prompt_embeddings",
                    prompt_embeddings,
                    persistent=False,
                )
        elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
            self.dinotxt_visual_model = self._load_dinotxt_visual_model()
        else:
            raise NotImplementedError('Feature extractor not implemented')

        if cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
            self.cross_motion_module = CrossMotionModule(
                out_feature_dim=self.embed_dim,
                num_patches=self.num_patches,
                in_fea_dim_crossmotion=2*self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE,
            )

        if cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE:
            if cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID:
                #TODO(pulkit): make this dynamic
                assert cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID_LEVELS == 3, 'hard coded for now'
                feat_dim = cfg.POINT_INFO.HOD.NUM_BINS * 7
            else:
                feat_dim = cfg.POINT_INFO.HOD.NUM_BINS
            self.hod_motion_module = HODMotionModule(
                in_feature_dim=feat_dim,
                out_feature_dim=self.embed_dim,
                num_patches=self.num_patches,
            )

    def hook_fn(self, feat_dict, layer_name):
        """Hook function to extract features of specific layers"""
        def hook(module, input, output):
            # Store the extracted features as an attribute of the model
            feat_dict[layer_name] = output
        return hook

    def backward_hook_fn(self, feat_dict, layer_name):
        """Backward hook function for extracting gradients"""
        def hook(module, grad_inputs, grad_outputs):
            # Store the extracted features as an attribute of the model
            grad_out_norm = np.mean([torch.norm(grad_output).item()
                                        for grad_output in grad_outputs])
            grad_in_norm = np.mean([torch.norm(grad_input).item()
                                        for grad_input in grad_inputs])
            feat_dict[layer_name] = {
                'grad_out': np.around(grad_out_norm, 3),
                'grad_in': np.around(grad_in_norm, 3)
            }
        return hook


    def init_weights(self):
        """Initialize weights"""
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """No weight decay params"""
        if self.cfg.MF.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        """Get classifier"""
        return self.head


    def get_point_grid_size(self):
        """Get point grid size"""
        all_divisors = divisors(self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)
        return all_divisors[len(all_divisors) // 2]

    def _resolve_prompt_bank_path(self, prompt_bank_path):
        """Resolve the prompt bank path for SAV text-cluster selection."""
        if prompt_bank_path:
            if os.path.isabs(prompt_bank_path):
                return prompt_bank_path
            return os.path.join(os.getcwd(), prompt_bank_path)
        default_path = os.path.join(os.getcwd(), "data", "sav", "clip_label_prompt_bank.json")
        if os.path.exists(default_path):
            return default_path
        raise FileNotFoundError(
            "Prompt bank path is not set and default SAV prompt bank was not found."
        )

    def _load_prompt_bank_embeddings(self, prompt_bank_path):
        """Load and encode the prompt bank into CLIP text space."""
        with open(prompt_bank_path, "r", encoding="utf-8") as handle:
            prompt_bank = json.load(handle)
        text_embeddings = []
        self.clip_model.eval()
        with torch.no_grad():
            for class_id in range(self.num_classes):
                prompts = prompt_bank.get(str(class_id))
                if not prompts:
                    raise KeyError(f"Missing prompt list for class id {class_id}")
                tokenized = self.clip_tokenize(prompts).cuda(non_blocking=True)
                prompt_features = self.clip_model.encode_text(tokenized).float()
                prompt_features = F.normalize(prompt_features, dim=-1)
                class_embedding = F.normalize(prompt_features.mean(dim=0), dim=-1)
                text_embeddings.append(class_embedding.unsqueeze(0))
        return torch.cat(text_embeddings, dim=0)

    def _torchhub_dirs(self):
        """Return candidate torch hub dirs for cached DINOv2 assets."""
        candidates = [
            Path(torch.hub.get_dir()),
            Path(os.getcwd()) / ".torch-cache" / "hub",
            Path.home() / ".cache" / "torch" / "hub",
        ]
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _find_torchhub_checkpoint(self, filename):
        """Find a non-empty checkpoint in known torch hub cache dirs."""
        for hub_dir in self._torchhub_dirs():
            checkpoint_path = hub_dir / "checkpoints" / filename
            if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
                return checkpoint_path
        return None

    def _find_dinov2_hub_repo(self):
        """Find or populate the cached DINOv2 hub repository."""
        for hub_dir in self._torchhub_dirs():
            hub_repo = hub_dir / "facebookresearch_dinov2_main"
            if hub_repo.exists():
                return hub_repo

        torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
            skip_validation=True,
        )
        for hub_dir in self._torchhub_dirs():
            hub_repo = hub_dir / "facebookresearch_dinov2_main"
            if hub_repo.exists():
                return hub_repo
        raise FileNotFoundError("Could not locate the cached DINOv2 hub repository.")

    def _load_dinotxt_visual_model(self):
        """Load the DinoTxt visual tower without loading the text encoder."""
        hub_repo = self._find_dinov2_hub_repo()
        hub_repo_str = str(hub_repo)
        if hub_repo_str not in sys.path:
            sys.path.insert(0, hub_repo_str)

        from dinov2.hub.backbones import dinov2_vitl14_reg
        from dinov2.hub.text.dinov2_wrapper import DINOv2Wrapper
        from dinov2.hub.text.vision_tower import VisionTower
        from dinov2.hub.utils import _DINOV2_BASE_URL

        backbone_checkpoint = self._find_torchhub_checkpoint(
            "dinov2_vitl14_reg4_pretrain.pth"
        )
        if backbone_checkpoint is not None:
            backbone = dinov2_vitl14_reg(weights=str(backbone_checkpoint))
        else:
            backbone = dinov2_vitl14_reg()

        visual_model = VisionTower(
            backbone=DINOv2Wrapper(backbone),
            freeze_backbone=True,
            embed_dim=2048,
            num_head_blocks=2,
            head_blocks_block_drop_path=0.3,
            use_class_token=True,
            use_patch_tokens=True,
            patch_token_layer=1,
            patch_tokens_pooler_type="mean",
            use_linear_projection=False,
        )

        vision_head_checkpoint = self._find_torchhub_checkpoint(
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth"
        )
        if vision_head_checkpoint is not None:
            vision_head_state_dict = torch.load(
                vision_head_checkpoint,
                map_location="cpu",
            )
        else:
            vision_head_state_dict = torch.hub.load_state_dict_from_url(
                _DINOV2_BASE_URL
                + "/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
                map_location="cpu",
            )

        visual_model.head.load_state_dict(vision_head_state_dict, strict=True)
        visual_model.cuda()
        visual_model.eval()
        for param in visual_model.parameters():
            param.requires_grad = False
        return visual_model

    def _sample_point_features(self, feat_to_use, pred_tracks, add_pt_pos_embed=False):
        """Sample point features from a dense patch feature map."""
        bs, num_frames = feat_to_use.shape[:2]
        feat_to_use = rearrange(feat_to_use, 'b t p q d -> (b t) p q d')
        feat_to_use = rearrange(feat_to_use, 'b p q d -> b d p q')
        num_x, num_y = feat_to_use.shape[-2:]
        assert self.num_patches == num_x * num_y, "Number of patches mismatch"
        pred_tracks = pred_tracks.view(bs * num_frames, -1, 1, 2)
        sampled_feat = F.grid_sample(
            feat_to_use,
            pred_tracks,
            align_corners=True,
            mode=self.cfg.MODEL.FEAT_EXTRACT_MODE,
        )
        if add_pt_pos_embed:
            spatial_pos_embed = self.spatial_pos_embed.repeat(bs * num_frames, 1, 1, 1)
            sample_pos_embedding = F.grid_sample(
                spatial_pos_embed,
                pred_tracks,
                align_corners=True,
                mode='bilinear',
            )
            sampled_feat = sampled_feat + sample_pos_embedding
        sampled_feat = rearrange(sampled_feat, 'b d p q -> b p q d')
        sampled_feat = sampled_feat.squeeze(-2)
        return rearrange(sampled_feat, '(b t) p d -> b t p d', t=num_frames)

    def _first_occurrence_mask(self, point_indices, valid_point_mask):
        """Keep the first copy of each repeated point index."""
        num_points = point_indices.numel()
        if num_points == 0:
            return torch.zeros_like(valid_point_mask, dtype=torch.bool)
        previous_points = torch.tril(
            torch.ones(
                (num_points, num_points),
                device=point_indices.device,
                dtype=torch.bool,
            ),
            diagonal=-1,
        )
        duplicate_previous = (
            (point_indices.unsqueeze(1) == point_indices.unsqueeze(0))
            & previous_points
            & valid_point_mask.unsqueeze(0)
        )
        return valid_point_mask & ~duplicate_previous.any(dim=1)

    def _masked_cluster_mean(self, point_features, point_mask, cluster_point_mask):
        """Average projected point features for one cluster over valid frames/points."""
        cluster_feat = point_features[:, cluster_point_mask, :]
        cluster_mask = point_mask[:, cluster_point_mask]
        weights = cluster_mask.float().unsqueeze(-1)
        denom = weights.sum().clamp_min(1.0)
        return (cluster_feat * weights).sum(dim=(0, 1)) / denom

    def _aggregate_cluster_repr(self, point_features, point_mask, point_clusters):
        """Aggregate projected point features into per-cluster representations."""
        point_weights = point_mask.float()
        point_feature_sum = (point_features * point_weights.unsqueeze(-1)).sum(dim=0)
        point_weight_sum = point_weights.sum(dim=0)
        cluster_ids, cluster_inverse = torch.unique(
            point_clusters,
            sorted=True,
            return_inverse=True,
        )
        cluster_feature_sum = point_feature_sum.new_zeros(
            (cluster_ids.shape[0], point_feature_sum.shape[-1])
        )
        cluster_feature_sum.index_add_(0, cluster_inverse, point_feature_sum)
        cluster_weight_sum = point_weight_sum.new_zeros(cluster_ids.shape[0])
        cluster_weight_sum.index_add_(0, cluster_inverse, point_weight_sum)
        cluster_repr = cluster_feature_sum / cluster_weight_sum.clamp_min(1.0).unsqueeze(-1)
        cluster_point_counts = torch.bincount(
            cluster_inverse,
            minlength=cluster_ids.shape[0],
        )
        return F.normalize(cluster_repr, dim=-1), cluster_ids, cluster_inverse, cluster_point_counts

    def _repeat_selected_points(self, branch_feature, branch_mask, point_weights):
        """Pad a selected branch back to the configured point count using repeated valid points."""
        points_to_sample = self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE
        valid_indices = torch.nonzero(branch_mask.any(dim=0), as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return None
        if valid_indices.numel() >= points_to_sample:
            indices_to_use = valid_indices[:points_to_sample]
        else:
            repeats_needed = points_to_sample - valid_indices.numel()
            repeat_indices = valid_indices[
                torch.arange(repeats_needed, device=valid_indices.device) % valid_indices.numel()
            ]
            indices_to_use = torch.cat([valid_indices, repeat_indices], dim=0)

        repeated_feature = branch_feature[:, :, indices_to_use, :]
        repeated_mask = branch_mask[:, indices_to_use]
        repeated_weights = point_weights[indices_to_use].clone()
        _, inverse_indices, counts = torch.unique(
            indices_to_use,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        repeated_weights = repeated_weights / counts[inverse_indices].to(repeated_weights.dtype)
        return repeated_feature, repeated_mask, repeated_weights

    def _build_support_conditioned_branches(
        self,
        appearance_feat,
        projected_point_feat,
        metadata,
        hod_motion_feat=None,
    ):
        """Build support-only label-conditioned branches for q2s few-shot matching."""
        support_mask = metadata['support_mask'].bool()
        episode_positive_labels = metadata['episode_positive_labels'].bool()
        base_pt_mask = (
            metadata['pred_query_mask']
            if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
            else metadata['pred_visibility']
        ).bool()
        obj_ids = metadata['obj_ids'].long()
        point_indices = metadata['point_indices'].long()
        episode_class_ids = metadata['episode_class_ids'].long()

        branch_features = []
        branch_masks = []
        branch_point_weights = []
        branch_class_indices = []
        branch_sample_indices = []
        support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
        for sample_idx in support_indices.tolist():
            sample_positive_labels = torch.nonzero(
                episode_positive_labels[sample_idx], as_tuple=False
            ).flatten()
            if sample_positive_labels.numel() == 0:
                continue

            sample_mask = base_pt_mask[sample_idx]
            valid_points = sample_mask.any(dim=0)
            unique_points = self._first_occurrence_mask(
                point_indices[sample_idx], valid_points
            )
            present_points = valid_points & unique_points
            sample_obj_ids = obj_ids[sample_idx]
            present_clusters = torch.unique(sample_obj_ids[present_points])
            if present_clusters.numel() == 0:
                continue

            present_point_feat = projected_point_feat[sample_idx][:, present_points, :]
            present_point_mask = sample_mask[:, present_points]
            present_cluster_ids = sample_obj_ids[present_points]
            cluster_repr, cluster_ids, cluster_inverse, cluster_point_counts = (
                self._aggregate_cluster_repr(
                    present_point_feat,
                    present_point_mask,
                    present_cluster_ids,
                )
            )

            sample_episode_class_ids = (
                episode_class_ids[sample_idx]
                if episode_class_ids.ndim == 2
                else episode_class_ids
            )
            global_class_indices = sample_episode_class_ids.index_select(0, sample_positive_labels)
            text_embeddings = self.text_cluster_prompt_embeddings.index_select(
                0,
                global_class_indices,
            )
            scores = torch.matmul(text_embeddings, cluster_repr.transpose(0, 1))
            top_k = min(self.text_cluster_cfg.TOP_M, scores.shape[1])
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
            cluster_weights = F.softmax(top_scores / self.text_cluster_cfg.TAU, dim=1)
            selected_cluster_ids = cluster_ids[top_indices]

            selected_point_mask = (
                sample_obj_ids.view(1, 1, -1) == selected_cluster_ids.unsqueeze(-1)
            ).any(dim=1)
            point_group_counts = torch.unique(
                point_indices[sample_idx],
                sorted=True,
                return_inverse=False,
                return_counts=True,
            )[1]
            _, point_group_inverse = torch.unique(
                point_indices[sample_idx],
                sorted=True,
                return_inverse=True,
                return_counts=False,
            )
            present_group_ids = point_group_inverse[present_points]
            selected_cluster_weights = cluster_weights / cluster_point_counts[top_indices].to(
                cluster_weights.dtype
            )
            weights_per_cluster = cluster_weights.new_zeros(
                (cluster_weights.shape[0], cluster_ids.shape[0])
            )
            weights_per_cluster.scatter_(1, top_indices, selected_cluster_weights)
            present_point_weights = weights_per_cluster.gather(
                1,
                cluster_inverse.unsqueeze(0).expand(weights_per_cluster.shape[0], -1),
            )
            group_weights = cluster_weights.new_zeros(
                (cluster_weights.shape[0], point_group_counts.shape[0])
            )
            group_weights.scatter_(
                1,
                present_group_ids.unsqueeze(0).expand(cluster_weights.shape[0], -1),
                present_point_weights,
            )
            point_weights = group_weights.gather(
                1,
                point_group_inverse.unsqueeze(0).expand(cluster_weights.shape[0], -1),
            )
            point_weights = point_weights / point_group_counts[point_group_inverse].unsqueeze(0).to(
                point_weights.dtype
            )

            branch_mask = sample_mask.unsqueeze(0) & selected_point_mask.unsqueeze(1)
            valid_branch_mask = branch_mask.reshape(branch_mask.shape[0], -1).any(dim=1)
            if not valid_branch_mask.any():
                continue

            sample_positive_labels = sample_positive_labels[valid_branch_mask]
            selected_point_mask = selected_point_mask[valid_branch_mask]
            point_weights = point_weights[valid_branch_mask]
            branch_mask = branch_mask[valid_branch_mask]

            branch_mask_float = branch_mask.unsqueeze(-1).to(appearance_feat.dtype)
            sample_appearance_feat = appearance_feat[sample_idx].unsqueeze(0)
            branch_feature = sample_appearance_feat * branch_mask_float
            if hod_motion_feat is not None:
                sample_hod_motion_feat = hod_motion_feat[sample_idx].unsqueeze(0)
                branch_feature = branch_feature + sample_hod_motion_feat * branch_mask_float
            if self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
                num_sample_branches = branch_mask.shape[0]
                pred_tracks_batch = metadata['pred_tracks'][sample_idx:sample_idx + 1].repeat(
                    num_sample_branches,
                    1,
                    1,
                    1,
                )
                pred_visibility_batch = metadata['pred_visibility'][
                    sample_idx:sample_idx + 1
                ].repeat(num_sample_branches, 1, 1)
                cross_motion_feat = self.cross_motion_module(
                    pred_tracks_batch,
                    pred_visibility_batch,
                    point_selection_mask=selected_point_mask,
                )
                branch_feature = branch_feature + cross_motion_feat * branch_mask_float

            for branch_idx in range(branch_mask.shape[0]):
                repeated_branch = self._repeat_selected_points(
                    branch_feature[branch_idx:branch_idx + 1],
                    branch_mask[branch_idx],
                    point_weights[branch_idx],
                )
                if repeated_branch is None:
                    continue
                repeated_feature, repeated_mask, repeated_weights = repeated_branch
                branch_features.append(repeated_feature)
                branch_masks.append(repeated_mask.unsqueeze(0))
                branch_point_weights.append(repeated_weights.unsqueeze(0))
                branch_class_indices.append(int(sample_positive_labels[branch_idx].item()))
                branch_sample_indices.append(sample_idx)

        if not branch_features:
            return None

        branch_feature_tensor = torch.cat(branch_features, dim=0)
        branch_mask_tensor = torch.cat(branch_masks, dim=0)
        _, branch_patch_tokens = self.pt_forward(
            branch_feature_tensor,
            {
                'pred_visibility': branch_mask_tensor,
                'pred_query_mask': branch_mask_tensor,
            },
        )
        return {
            'support_conditioned_patch_tokens': branch_patch_tokens,
            'support_branch_point_weights': torch.cat(branch_point_weights, dim=0),
            'support_branch_class_indices': torch.tensor(
                branch_class_indices,
                device=branch_patch_tokens.device,
                dtype=torch.long,
            ),
            'support_branch_sample_indices': torch.tensor(
                branch_sample_indices,
                device=branch_patch_tokens.device,
                dtype=torch.long,
            ),
        }


    def get_dino_features(self, x):
        """ Get DINO features
        Args:
            x (torch.Tensor): Input features of shape [BS, T, C, H, W]

        Returns:
            torch.Tensor: DINO features of shape [BS, T, P, Q, D]
        """
        self.dino.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            _ = self.dino(x)
        else:
            with torch.no_grad():
                _ = self.dino(x)
        #using hooks to get the patch tokens
        feat = self.feat_dict['dino'][:, self.dino.num_register_tokens + 1 :]
        feat_size = feat.shape[-1]
        #dino patch side is fine
        feat = feat.view(batch_size, num_frames, self.patch_num_side,
                         self.patch_num_side, feat_size)

        return feat

    def get_clip_features(self, x, return_projected=False):
        """Get CLIP ViT-B/16 patch features."""
        self.clip_visual.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        x = x.type(self.clip_visual.conv1.weight.dtype)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            feat = self._forward_clip_visual(x, return_projected=return_projected)
        else:
            with torch.no_grad():
                feat = self._forward_clip_visual(x, return_projected=return_projected)
        if return_projected:
            raw_feat, projected_feat = feat
            raw_feat = raw_feat.float().view(
                batch_size,
                num_frames,
                self.patch_num_side,
                self.patch_num_side,
                raw_feat.shape[-1],
            )
            projected_feat = projected_feat.float().view(
                batch_size,
                num_frames,
                self.patch_num_side,
                self.patch_num_side,
                projected_feat.shape[-1],
            )
            return raw_feat, projected_feat
        feat = feat.float()
        feat = feat.view(
            batch_size,
            num_frames,
            self.patch_num_side,
            self.patch_num_side,
            feat.shape[-1],
        )
        return feat

    def _forward_clip_visual(self, x, return_projected=False):
        """Return raw CLIP patch tokens before ln_post/proj."""
        visual = self.clip_visual
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        raw_patch = x[:, 1:, :]
        if not return_projected:
            return raw_patch
        projected_patch = (visual.ln_post(x) @ visual.proj)[:, 1:, :]
        return raw_patch, projected_patch

    def get_dinotxt_features(self, x):
        """Get DinoTxt ViT-L/14 visual patch features."""
        self.dinotxt_visual_model.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            _, patch_tokens = self.dinotxt_visual_model.get_class_and_patch_tokens(x)
        else:
            with torch.no_grad():
                _, patch_tokens = self.dinotxt_visual_model.get_class_and_patch_tokens(x)
        patch_tokens = patch_tokens.float()
        return patch_tokens.view(
            batch_size,
            num_frames,
            self.patch_num_side,
            self.patch_num_side,
            patch_tokens.shape[-1],
        )


    def pt_forward(self, x, metadata):
        """ Forward pass for point tracking based transformer model.

        Args:
            x (torch.Tensor): Input features of shape [BS, T, N, D]
            metadata (dict): Metadata containing prediction masks

        Returns:
            torch.Tensor: Class token features
            torch.Tensor: Patch token features
        """
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK:
            pt_mask = metadata['pred_query_mask'] # [BS, T, N]
        else:
            pt_mask = metadata['pred_visibility'] # [BS, T, N]

        bs, temporal_dim, num_points, _ = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t n d -> b n t d')
        pt_mask = rearrange(pt_mask, 'b t n -> b n t')
        x = rearrange(x, 'b n t d -> b (n t) d')
        pt_mask = rearrange(pt_mask, 'b n t -> b (n t)')
        if self.cfg.MODEL.USE_CLS_TOKEN:
            cls_tokens = self.cls_token.expand(bs, -1, -1) # [BS, 1, dim]
            x = torch.cat((cls_tokens, x), dim=1) # [BS, N, dim]
            cls_token_mask = torch.ones(bs, 1).bool().to(x.device)
            pt_mask = torch.cat((cls_token_mask, pt_mask), dim=1) # [BS, N, dim]
        # Apply positional dropout
        x = self.pos_drop(x) # [BS, N, dim]
        # Encoding using transformer layers
        thw = [self.temporal_resolution, self.point_grid_size,
            int(num_points / self.point_grid_size)]
        for _, blk in enumerate(self.blocks):
            x, _ = blk(
                x,
                thw,
                pt_mask
            )
        if self.cfg.MODEL.ADAPOOLING.ENABLE:
            if self.cfg.MODEL.ADAPOOLING.TYPE == 'temporal_spatial':
                extra_cls_token =  self.agg_cls_token.expand(bs * num_points, -1, -1)
                cls_x, patch_x = self.adaptive_pooling(x, extra_cls_token)
                patch_x = rearrange(patch_x, 'b n d -> b 1 n d')

            elif self.cfg.MODEL.ADAPOOLING.TYPE == 'spatial_temporal':
                extra_cls_token = self.agg_cls_token.expand(bs * temporal_dim, -1, -1)
                cls_x, patch_x = self.adaptive_pooling(x, extra_cls_token)
                patch_x = rearrange(patch_x, 'b t d -> b t 1 d')

        else:
            x = self.norm(x)
            if self.cfg.MODEL.USE_CLS_TOKEN:
                cls_x, patch_x = x[:, 0], x[:, 1:]
                if self.cfg.MODEL.USE_PATCH_AS_CLS:
                    cls_x = patch_x.mean(dim=1)
            else:
                # If cls token is not ued, for now using global average pooling
                cls_x = x.mean(dim=1)
                patch_x = x

            cls_x = self.pre_logits(cls_x)
            # Taking the patch tokens back to the input shape
            patch_x = rearrange(patch_x, 'b (n t) d -> b t n d', t=temporal_dim)
        if not torch.isfinite(x).all():
            print("WARNING: nan in features out")
        return cls_x, patch_x

    def add_st_pos_embeddings(self, x):
        """ Add spatial and temporal positional embeddings to the input features.

        Args:
            x (torch.Tensor): Input features of shape [BS, T, P, Q, D]

        Returns:
            torch.Tensor: Output features of shape [BS, T, P, Q, D]
        """
        _, _, sp_dim_1, sp_dim_2, _ = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t p q d -> b t (p q) d')
        x = x + self.space_pos_embed.unsqueeze(0)
        x = self.space_pos_drop(x)
        x = rearrange(x, 'b t p d -> b p t d')
        x = x + self.time_pos_embed.unsqueeze(0)
        x = self.time_pos_drop(x)
        x = rearrange(x, 'b (p q) t d -> b t p q d', p=sp_dim_1, q=sp_dim_2)
        return x

    def forward(self, input_to_use):
        """Forward pass of the model"""
        x = input_to_use['video']
        metadata = input_to_use['metadata']
        few_shot_aux = None

        if 'skip_feat_extractor' in input_to_use:
            skip_feat_extractor = input_to_use['skip_feat_extractor']
        else:
            skip_feat_extractor = False
        projected_feat_to_use = None
        if not self.cfg.MODEL.APPEARANCE_MODULE_DISABLE:
            if skip_feat_extractor:
                embed_dim = self.embed_dim
                batch_size, num_frames = x.shape[:2]
                feat_to_use = torch.randn(batch_size, num_frames,
                                            int(self.num_patches**0.5),
                                            int(self.num_patches**0.5),
                                            embed_dim).to(x.device)
            else:
                if self.feat_extractor_type == "dino":
                    feat_to_use = self.get_dino_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        # for ablation study without point tracking module
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)
                elif self.feat_extractor_type == "clip_vit_b16":
                    if self.use_text_conditioned_support:
                        feat_to_use, projected_feat_to_use = self.get_clip_features(
                            x,
                            return_projected=True,
                        )
                    else:
                        feat_to_use = self.get_clip_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)
                elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
                    feat_to_use = self.get_dinotxt_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)
                else:
                    raise NotImplementedError('Feature extractor not implemented')

            if self.cfg.MF.USE_BASE_POS_EMBED:
                feat_to_use = self.add_st_pos_embeddings(feat_to_use)


            if self.cfg.POINT_INFO.ENABLE:
                pred_tracks = metadata['pred_tracks']
                sampled_feat = self._sample_point_features(
                    feat_to_use,
                    pred_tracks,
                    add_pt_pos_embed=(
                        self.cfg.MF.USE_PT_SPACE_POS_EMBED
                        and self.cfg.FEW_SHOT.USE_MODEL
                        and not self.cfg.MF.USE_BASE_POS_EMBED
                    ),
                )
                if projected_feat_to_use is not None:
                    projected_point_feat = self._sample_point_features(
                        projected_feat_to_use,
                        pred_tracks,
                        add_pt_pos_embed=False,
                    )
                else:
                    projected_point_feat = None

            else:
                sampled_feat = rearrange(feat_to_use, 'b t p q d -> b t (p q) d')
                self.point_grid_size = int(sampled_feat.shape[2] ** 0.5)
                projected_point_feat = None
        else:
            sampled_feat = 0
            projected_point_feat = None

        appearance_feat = sampled_feat
        hod_motion_feat = None
        if self.cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE:
            hod_motion_feat = self.hod_motion_module(metadata['hod_feat'].float())
            sampled_feat = sampled_feat + hod_motion_feat

        if self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
            cross_motion_feat = self.cross_motion_module(
                metadata['pred_tracks'], metadata['pred_visibility'])
            sampled_feat = sampled_feat + cross_motion_feat

        cls_x, patch_x = self.pt_forward(sampled_feat, metadata)
        if (
            self.use_text_conditioned_support
            and projected_point_feat is not None
            and 'support_mask' in metadata
            and 'episode_positive_labels' in metadata
            and 'obj_ids' in metadata
            and 'point_indices' in metadata
        ):
            few_shot_aux = self._build_support_conditioned_branches(
                appearance_feat,
                projected_point_feat,
                metadata,
                hod_motion_feat=hod_motion_feat,
            )
        # x = self.forward_features(x, metadata) # [BS, d]
        x = self.head_drop(cls_x)


        x = self.head(x)
        # previously there was a softmax here for validation which messed up the loss computation
        if self.cfg.TASK == 'few_shot':
            return x, patch_x, few_shot_aux
        return x


def compute_correlation_map(features):
    """
    Compute correlation map between consecutive frames
    Args:
        features: Tensor of shape [bs, num_frames, num_patches, d]
    Returns:
        correlation: Tensor of shape [bs, num_frames-1, num_patches, num_patches]
    """
    bs, num_frames, num_patches, d = features.shape

    # Get features for all frames except last one
    feat1 = features[:, :-1]  # [bs, num_frames-1, num_patches, d]

    # Get features for all frames except first one
    feat2 = features[:, 1:]   # [bs, num_frames-1, num_patches, d]

    # Normalize the features
    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)

    # Compute correlation through matrix multiplication
    # Reshape for batch matrix multiplication
    feat1_reshaped = feat1_norm.view(bs * (num_frames-1), num_patches, d)
    feat2_reshaped = feat2_norm.view(bs * (num_frames-1), num_patches, d)

    # Compute correlation [bs*(num_frames-1), num_patches, num_patches]
    correlation = torch.bmm(feat1_reshaped, feat2_reshaped.transpose(1, 2))

    # Reshape back to [bs, num_frames-1, num_patches, num_patches]
    correlation = correlation.view(bs, num_frames-1, num_patches, num_patches)

    return correlation


def get_points_using_correlation(cfg, features):
    """
    Get points using correlation
    Args:
        cfg (dict): Configuration dictionary
        features (torch.Tensor): Features of shape [BS, T, P, Q, D]

    Returns:
        dict: Metadata containing prediction tracks and visibility
    """
    features = rearrange(features, 'b t p q d -> b t (p q) d')
    bs, num_frames, num_patches, _ = features.shape

    # Compute basic correlation first
    correlation = compute_correlation_map(features)

    # Apply mutual matching
    corr_b = correlation.view(bs*(num_frames-1), num_patches, num_patches)
    corr_a = corr_b

    # Get max values
    corr_b_max, _ = torch.max(corr_b, dim=1, keepdim=True)
    corr_a_max, _ = torch.max(corr_a, dim=2, keepdim=True)

    # Normalize by max values
    eps = 1e-5
    corr_b = corr_b / (corr_b_max + eps)
    corr_a = corr_a / (corr_a_max + eps)

    # Compute mutual correlation
    mutual_correlation = correlation * (corr_a.view_as(correlation) * corr_b.view_as(correlation))
    _, max_indices = torch.max(mutual_correlation, dim=-1)
    extra_indices = torch.arange(max_indices.shape[-1]).to(max_indices.device)
    extra_indices = extra_indices.unsqueeze(0).unsqueeze(0).expand(max_indices.shape[0], -1, -1)
    max_indices = torch.cat([extra_indices, max_indices], dim=1)
    if cfg.MODEL.FEAT_EXTRACTOR == "dino" and cfg.MODEL.DINO_CONFIG == "dinov2_vitb14":
        grid_points = create_normalized_grid(image_size=224, grid_size=16)
    elif cfg.MODEL.FEAT_EXTRACTOR == "clip_vit_b16":
        grid_points = create_normalized_grid(image_size=224, grid_size=14)
    elif cfg.MODEL.FEAT_EXTRACTOR == "dinotxt_vitl14_reg4":
        grid_points = create_normalized_grid(image_size=224, grid_size=16)
    else:
        raise NotImplementedError(
            f'Grid points dim not set for extractor {cfg.MODEL.FEAT_EXTRACTOR}'
        )
    grid_points = grid_points.to(max_indices.device)
    grid_points = rearrange(grid_points, 'n d -> 1 1 n d')
    grid_points = repeat(grid_points, '1 1 n d -> b t n d', b=bs, t=num_frames)
    new_metadata = sample_grid_points_with_indices(cfg, grid_points, max_indices)

    return new_metadata

def create_normalized_grid(image_size=224, grid_size=16):
    """
    Create a normalized grid of points from an image
    Args:
        image_size (int): Size of the image (assuming square image)
        grid_size (int): Size of the grid (e.g., 16 for 16x16 grid)
    Returns:
        grid_points: Tensor of shape [grid_size*grid_size, 2] containing normalized x,y coordinates
    """
    # Create linear spaces for x and y coordinates
    grid_step = image_size // grid_size
    points = torch.arange(grid_step // 2, image_size, grid_step)

    # Create meshgrid
    y, x = torch.meshgrid(points, points, indexing='ij')

    # Reshape to [N, 2] where N = grid_size * grid_size
    grid_points = torch.stack([x, y], dim=-1).reshape(-1, 2)

    # Normalize to [-1, 1]
    grid_points = 2 * (grid_points / (image_size - 1)) - 1

    return grid_points


def sample_grid_points_with_indices(cfg, grid_points, max_indices):
    """
    Sample grid points using max correlation indices
    Args:
        grid_points: Tensor of shape [bs, num_frames, num_patches, 2]
        max_indices: Tensor of shape [bs, num_frames, num_patches]
    Returns:
        sampled_points: Tensor of shape [bs, num_frames, num_patches, 2]
    """
    bs, num_frames, num_patches, _ = grid_points.shape

    # Create batch indices for gathering
    batch_indices = torch.arange(bs).view(-1, 1, 1).expand(-1, num_frames, num_patches)
    frame_indices = torch.arange(num_frames).view(1, -1, 1).expand(bs, -1, num_patches)

    # Gather points using indices
    sampled_points = grid_points[batch_indices, frame_indices, max_indices]
    new_metadata = {}
    if cfg.POINT_INFO.HOD.GET_FEAT:
        sampled_points_for_hod = rearrange(sampled_points, 'b t n d -> b n t d')
        hod_feats = []
        for i in range(bs):
            hod_feat = torch.tensor(get_orientation_hist(
                                        sampled_points_for_hod[i].cpu().numpy(),
                                        cfg.POINT_INFO.HOD.NUM_BINS,
                                        preserve_temporal=True))
            hod_feats.append(hod_feat.unsqueeze(0))
        hod_feats = torch.cat(hod_feats, dim=0).to(grid_points.device)
        new_metadata['hod_feat'] = hod_feats
    new_metadata['pred_tracks'] = sampled_points
    new_metadata['pred_visibility'] = torch.ones_like(
                            sampled_points).bool()[...,0].to(grid_points.device)
    new_metadata['pred_query_mask'] = torch.ones_like(
                            sampled_points).bool()[...,0].to(grid_points.device)


    return new_metadata
