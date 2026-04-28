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
            and self.feat_extractor_type in ("clip_vit_b16", "dinotxt_vitl14_reg4")
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
        if self.use_text_conditioned_support:
            if self.feat_extractor_type == "clip_vit_b16":
                self.text_feature_dim = 512
            elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
                self.text_feature_dim = self.embed_dim
            else:
                raise NotImplementedError(
                    f"Text support branch is not supported for {self.feat_extractor_type}."
                )
            if self.text_feature_dim == self.embed_dim:
                self.text_to_model_proj = nn.Identity()
            else:
                self.text_to_model_proj = nn.Linear(self.text_feature_dim, self.embed_dim)
            self.text_gate_mlp = nn.Sequential(
                norm_layer(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.text_inject_alpha = nn.Parameter(torch.tensor(0.1))
            self.text_inject_beta = nn.Parameter(torch.tensor(0.0))
            self.use_soft_label_route = cfg.FEW_SHOT.SOFT_LABEL_ROUTE.ENABLE
            if self.use_soft_label_route:
                self.label_q_mod_mlp = nn.Sequential(
                    norm_layer(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim * 2),
                )
                self.label_slot_q = nn.Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)
                self.label_slot_kv = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=self.qkv_bias)
                self.label_slot_proj = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.use_soft_label_route = False

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
                prompt_embeddings, prompt_mask = self._load_prompt_bank_embeddings(
                    prompt_bank_path
                )
                self.register_buffer(
                    "text_prompt_embeddings",
                    prompt_embeddings,
                    persistent=False,
                )
                self.register_buffer(
                    "text_prompt_mask",
                    prompt_mask,
                    persistent=False,
                )
        elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
            self.dinotxt_visual_model = self._load_dinotxt_visual_model()
            if self.use_text_conditioned_support:
                self.dinotxt_text_model = self._load_dinotxt_text_model()
                self.dinotxt_tokenizer = self._load_dinotxt_tokenizer()
                prompt_bank_path = self._resolve_prompt_bank_path(
                    self.text_cluster_cfg.PROMPT_BANK_PATH
                )
                prompt_embeddings, prompt_mask = self._load_prompt_bank_embeddings(
                    prompt_bank_path
                )
                self.register_buffer(
                    "text_prompt_embeddings",
                    prompt_embeddings,
                    persistent=False,
                )
                self.register_buffer(
                    "text_prompt_mask",
                    prompt_mask,
                    persistent=False,
                )
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

    def _load_prompt_bank(self, prompt_bank_path):
        """Load the class prompt bank for text-conditioned cluster selection."""
        with open(prompt_bank_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_prompt_bank_embeddings(self, prompt_bank_path):
        """Load and encode prompt text as per-class prompt sequences."""
        if self.feat_extractor_type == "clip_vit_b16":
            return self._load_clip_prompt_bank_embeddings(prompt_bank_path)
        if self.feat_extractor_type == "dinotxt_vitl14_reg4":
            return self._load_dinotxt_prompt_bank_embeddings(prompt_bank_path)
        raise NotImplementedError(
            f"Text cluster is not supported for {self.feat_extractor_type}."
        )

    def _pad_prompt_embeddings(self, per_class_embeddings):
        """Pad variable-length prompt lists into one tensor plus a bool mask."""
        max_prompts = max(embeddings.shape[0] for embeddings in per_class_embeddings)
        feat_dim = per_class_embeddings[0].shape[-1]
        padded = per_class_embeddings[0].new_zeros(
            (len(per_class_embeddings), max_prompts, feat_dim)
        )
        mask = torch.zeros(
            (len(per_class_embeddings), max_prompts),
            device=padded.device,
            dtype=torch.bool,
        )
        for class_id, embeddings in enumerate(per_class_embeddings):
            num_prompts = embeddings.shape[0]
            padded[class_id, :num_prompts] = embeddings
            mask[class_id, :num_prompts] = True
        return padded, mask

    def _load_clip_prompt_bank_embeddings(self, prompt_bank_path):
        """Load and encode the prompt bank into CLIP text space."""
        prompt_bank = self._load_prompt_bank(prompt_bank_path)
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
                text_embeddings.append(prompt_features)
        return self._pad_prompt_embeddings(text_embeddings)

    def _load_dinotxt_prompt_bank_embeddings(self, prompt_bank_path):
        """Load and encode the prompt bank into DinoTxt patch text space."""
        prompt_bank = self._load_prompt_bank(prompt_bank_path)
        text_embeddings = []
        self.dinotxt_text_model.eval()
        with torch.no_grad():
            for class_id in range(self.num_classes):
                prompts = prompt_bank.get(str(class_id))
                if not prompts:
                    raise KeyError(f"Missing prompt list for class id {class_id}")
                tokenized = self.dinotxt_tokenizer.tokenize(prompts).cuda(non_blocking=True)
                prompt_features = self.dinotxt_text_model(tokenized).float()
                patch_prompt_features = prompt_features[:, prompt_features.shape[-1] // 2 :]
                patch_prompt_features = F.normalize(patch_prompt_features, dim=-1)
                text_embeddings.append(patch_prompt_features)
        return self._pad_prompt_embeddings(text_embeddings)

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

    def _get_dinov2_hub_modules(self):
        """Make DINOv2 hub modules importable and return the hub repo path."""
        hub_repo = self._find_dinov2_hub_repo()
        hub_repo_str = str(hub_repo)
        if hub_repo_str not in sys.path:
            sys.path.insert(0, hub_repo_str)
        return hub_repo

    def _download_torchhub_checkpoint(self, filename, url):
        """Download a checkpoint-like asset into the active torch hub cache."""
        target_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(target_path), progress=True)
        return target_path

    def _load_dinotxt_visual_model(self):
        """Load the DinoTxt visual tower without loading the text encoder."""
        self._get_dinov2_hub_modules()

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

    def _load_dinotxt_text_model(self):
        """Load the DinoTxt text tower for prompt embeddings."""
        self._get_dinov2_hub_modules()

        from dinov2.hub.text.text_tower import TextTower
        from dinov2.hub.text.text_transformer import TextTransformer
        from dinov2.hub.utils import _DINOV2_BASE_URL

        text_backbone = TextTransformer(
            context_length=77,
            vocab_size=49408,
            dim=1280,
            num_heads=20,
            num_layers=24,
            ffn_ratio=4,
            is_causal=True,
            ls_init_value=None,
            dropout_prob=0.0,
        )
        text_model = TextTower(
            backbone=text_backbone,
            freeze_backbone=False,
            embed_dim=2048,
            num_head_blocks=0,
            head_blocks_is_causal=False,
            head_blocks_block_drop_prob=0.0,
            tokens_pooler_type="argmax",
            use_linear_projection=True,
        )

        text_checkpoint = self._find_torchhub_checkpoint(
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth"
        )
        if text_checkpoint is not None:
            text_state_dict = torch.load(text_checkpoint, map_location="cpu")
        else:
            text_state_dict = torch.hub.load_state_dict_from_url(
                _DINOV2_BASE_URL
                + "/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
                map_location="cpu",
            )

        text_model.load_state_dict(text_state_dict, strict=True)
        text_model.cuda()
        text_model.eval()
        for param in text_model.parameters():
            param.requires_grad = False
        return text_model

    def _load_dinotxt_tokenizer(self):
        """Load the DinoTxt tokenizer, caching the BPE vocabulary if needed."""
        self._get_dinov2_hub_modules()

        from dinov2.hub.text.tokenizer import Tokenizer
        from dinov2.hub.utils import _DINOV2_BASE_URL

        vocab_filename = "bpe_simple_vocab_16e6.txt.gz"
        vocab_path = self._find_torchhub_checkpoint(vocab_filename)
        if vocab_path is None:
            vocab_path = self._download_torchhub_checkpoint(
                vocab_filename,
                _DINOV2_BASE_URL + f"/thirdparty/{vocab_filename}",
            )
        return Tokenizer(vocab_path=str(vocab_path))

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

    def _build_support_conditioned_branches(
        self,
        fused_feat,
        metadata,
    ):
        """Build support-only label-conditioned branches for q2s few-shot matching."""
        support_mask = metadata['support_mask'].bool()
        episode_positive_labels = metadata['episode_positive_labels'].bool()
        base_pt_mask = (
            metadata['pred_query_mask']
            if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
            else metadata['pred_visibility']
        ).bool()
        episode_class_ids = metadata['episode_class_ids'].long()

        branch_features = []
        branch_masks = []
        branch_class_indices = []
        branch_global_class_indices = []
        branch_sample_indices = []
        support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
        for sample_idx in support_indices.tolist():
            sample_positive_labels = torch.nonzero(
                episode_positive_labels[sample_idx], as_tuple=False
            ).flatten()
            if sample_positive_labels.numel() == 0:
                continue

            sample_episode_class_ids = (
                episode_class_ids[sample_idx]
                if episode_class_ids.ndim == 2
                else episode_class_ids
            )
            global_class_indices = sample_episode_class_ids.index_select(0, sample_positive_labels)
            num_sample_branches = sample_positive_labels.shape[0]
            branch_features.append(
                fused_feat[sample_idx:sample_idx + 1].repeat(
                    num_sample_branches,
                    1,
                    1,
                    1,
                )
            )
            branch_masks.append(
                base_pt_mask[sample_idx:sample_idx + 1].repeat(
                    num_sample_branches,
                    1,
                    1,
                )
            )
            branch_class_indices.append(sample_positive_labels)
            branch_global_class_indices.append(global_class_indices)
            branch_sample_indices.extend([sample_idx] * num_sample_branches)

        if not branch_features:
            return None

        branch_feature_tensor = torch.cat(branch_features, dim=0)
        branch_mask_tensor = torch.cat(branch_masks, dim=0)
        branch_class_indices = torch.cat(branch_class_indices, dim=0)
        branch_global_class_indices = torch.cat(branch_global_class_indices, dim=0)
        text_tokens, text_mask, text_global = self._get_branch_text_features(
            branch_global_class_indices,
            branch_feature_tensor.dtype,
        )
        branch_feature_tensor = self._inject_text_condition(
            branch_feature_tensor,
            text_global,
        )
        _, branch_patch_tokens = self.text_conditioned_pt_forward(
            branch_feature_tensor,
            {
                'pred_visibility': branch_mask_tensor,
                'pred_query_mask': branch_mask_tensor,
            },
            text_tokens,
            text_mask,
        )
        return {
            'support_conditioned_patch_tokens': branch_patch_tokens,
            'support_branch_point_weights': branch_patch_tokens.new_ones(
                (branch_patch_tokens.shape[0], branch_patch_tokens.shape[2])
            ),
            'support_branch_class_indices': branch_class_indices.to(
                device=branch_patch_tokens.device,
                dtype=torch.long,
            ),
            'support_branch_sample_indices': torch.tensor(
                branch_sample_indices,
                device=branch_patch_tokens.device,
                dtype=torch.long,
            ),
        }

    def _otsu_threshold(self, scores):
        """Compute a detached Otsu threshold for one trajectory-label score vector."""
        scores_detached = torch.nan_to_num(
            scores.detach().float(),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        if scores_detached.numel() == 0:
            return scores.new_tensor(0.0)
        if torch.isclose(scores_detached.max(), scores_detached.min()):
            return scores_detached.mean().to(device=scores.device, dtype=scores.dtype)

        route_cfg = self.cfg.FEW_SHOT.SOFT_LABEL_ROUTE
        num_bins = max(int(getattr(route_cfg, "OTSU_BINS", 64)), 2)
        hist = torch.histc(scores_detached, bins=num_bins, min=-1.0, max=1.0)
        hist_sum = hist.sum()
        if hist_sum <= 0:
            return scores_detached.mean().to(device=scores.device, dtype=scores.dtype)

        bin_width = 2.0 / num_bins
        bin_centers = torch.linspace(
            -1.0 + 0.5 * bin_width,
            1.0 - 0.5 * bin_width,
            num_bins,
            device=scores_detached.device,
            dtype=scores_detached.dtype,
        )
        prob = hist / hist_sum.clamp_min(1.0)
        omega = prob.cumsum(dim=0)
        mu = (prob * bin_centers).cumsum(dim=0)
        mu_total = mu[-1]
        denom = omega * (1.0 - omega)
        valid = denom > 1e-12
        if not bool(valid.any().item()):
            return scores_detached.mean().to(device=scores.device, dtype=scores.dtype)

        sigma_b = (mu_total * omega - mu).pow(2) / denom.clamp_min(1e-12)
        sigma_b = sigma_b.masked_fill(~valid, -1.0)
        threshold = bin_centers[sigma_b.argmax()]
        return threshold.to(device=scores.device, dtype=scores.dtype)

    def _compute_otsu_gate(self, traj_repr, label_text):
        """Compute one label's trajectory relevance scores, Otsu threshold, and gate."""
        traj_repr = torch.nan_to_num(traj_repr, nan=0.0, posinf=0.0, neginf=0.0)
        label_text = torch.nan_to_num(label_text, nan=0.0, posinf=0.0, neginf=0.0)
        if label_text.ndim == 1:
            label_text = label_text.unsqueeze(0)
        traj_repr = F.normalize(traj_repr, dim=-1)
        label_text = F.normalize(label_text, dim=-1)
        traj_repr = torch.nan_to_num(traj_repr, nan=0.0, posinf=0.0, neginf=0.0)
        label_text = torch.nan_to_num(label_text, nan=0.0, posinf=0.0, neginf=0.0)

        scores = torch.matmul(traj_repr, label_text.squeeze(0))
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        threshold = self._otsu_threshold(scores).detach()
        route_cfg = self.cfg.FEW_SHOT.SOFT_LABEL_ROUTE
        temperature = max(float(getattr(route_cfg, "GATE_TEMPERATURE", 0.05)), 1e-6)
        gate = torch.sigmoid((scores - threshold) / temperature)
        gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return scores, threshold, gate

    def _text_relation_spatial_forward(self, x, label_text, gate, point_mask):
        """Run standard temporal attention and text-gated relation spatial attention."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if label_text.ndim == 1:
            label_text = label_text.unsqueeze(0)
        if gate.ndim == 1:
            gate = gate.unsqueeze(0)

        bs, temporal_dim, num_points, _ = x.shape
        if point_mask is None:
            point_mask = torch.ones(
                bs,
                temporal_dim,
                num_points,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.bool()

        x = rearrange(x, 'b t n d -> b n t d')
        point_mask = rearrange(point_mask, 'b t n -> b n t')
        x = rearrange(x, 'b n t d -> b (n t) d')
        point_mask = rearrange(point_mask, 'b n t -> b (n t)')
        if self.cfg.MODEL.USE_CLS_TOKEN:
            cls_tokens = self.cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            cls_token_mask = torch.ones(bs, 1, device=x.device, dtype=torch.bool)
            point_mask = torch.cat((cls_token_mask, point_mask), dim=1)

        text_mod = self.label_q_mod_mlp(label_text.to(dtype=x.dtype, device=x.device))
        gamma, beta = text_mod.chunk(2, dim=-1)
        gate = gate.to(device=x.device, dtype=x.dtype)
        thw = [
            temporal_dim,
            self.point_grid_size,
            int(num_points / self.point_grid_size),
        ]
        relation_eps = float(getattr(self.cfg.FEW_SHOT.SOFT_LABEL_ROUTE, "RELATION_EPS", 1e-6))
        for _, blk in enumerate(self.blocks):
            x, _ = blk.forward_text_relation_conditioned(
                x,
                thw,
                point_mask,
                (gamma, beta),
                gate,
                relation_eps,
            )

        x = self.norm(x)
        if self.cfg.MODEL.USE_CLS_TOKEN:
            patch_x = x[:, 1:]
        else:
            patch_x = x
        patch_x = rearrange(patch_x, 'b (n t) d -> b t n d', t=temporal_dim)
        return torch.nan_to_num(patch_x, nan=0.0, posinf=0.0, neginf=0.0)

    def _label_slot_prototypes(self, h_st, label_text):
        """Use label text slots to aggregate all trajectory-time tokens."""
        h_st = torch.nan_to_num(h_st, nan=0.0, posinf=0.0, neginf=0.0)
        label_text = torch.nan_to_num(label_text, nan=0.0, posinf=0.0, neginf=0.0)
        bs, num_frames, num_points, feat_dim = h_st.shape
        num_labels = label_text.shape[1]
        tokens = rearrange(h_st, 'b t m c -> b (t m) c')
        orig_dtype = tokens.dtype
        query = self.label_slot_q(label_text)
        kv = self.label_slot_kv(tokens)
        q = rearrange(query, 'b z (h d) -> b h z d', h=self.num_heads)
        kv = rearrange(kv, 'b n (two h d) -> two b h n d', two=2, h=self.num_heads)
        k, v = kv[0], kv[1]
        q = q.float()
        k = k.float()
        v = v.float()
        scale = (feat_dim // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = F.softmax(attn / self.cfg.FEW_SHOT.SOFT_LABEL_ROUTE.SLOT_TAU, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h z d -> b z (h d)')
        out = self.label_slot_proj(out.to(orig_dtype))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_soft_label_routed_support_prototypes(self, fused_feat, metadata):
        """Build per-label Otsu relation-gated support prototypes."""
        support_mask = metadata['support_mask'].bool()
        episode_positive_labels = metadata['episode_positive_labels'].bool()
        base_pt_mask = (
            metadata['pred_query_mask']
            if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
            else metadata['pred_visibility']
        ).bool()
        episode_class_ids = metadata['episode_class_ids'].long()

        slot_tokens = []
        slot_class_indices = []
        slot_sample_indices = []
        support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
        for sample_idx in support_indices.tolist():
            positive_labels = torch.nonzero(
                episode_positive_labels[sample_idx],
                as_tuple=False,
            ).flatten()
            if positive_labels.numel() == 0:
                continue

            sample_episode_class_ids = (
                episode_class_ids[sample_idx]
                if episode_class_ids.ndim == 2
                else episode_class_ids
            )
            global_class_indices = sample_episode_class_ids.index_select(0, positive_labels)
            _, _, label_text = self._get_branch_text_features(
                global_class_indices,
                fused_feat.dtype,
            )

            support_feat = fused_feat[sample_idx:sample_idx + 1]
            support_mask_i = base_pt_mask[sample_idx:sample_idx + 1]
            traj_repr = support_feat.mean(dim=1).squeeze(0)

            for label_offset, positive_label in enumerate(positive_labels):
                label_text_i = label_text[label_offset]
                _, _, gate = self._compute_otsu_gate(traj_repr, label_text_i)
                h_st = self._text_relation_spatial_forward(
                    support_feat,
                    label_text_i.unsqueeze(0),
                    gate.unsqueeze(0),
                    support_mask_i,
                )
                sample_slot = self._label_slot_prototypes(
                    h_st,
                    label_text_i.view(1, 1, -1),
                ).squeeze(0)
                sample_slot = torch.nan_to_num(
                    sample_slot,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                slot_tokens.append(sample_slot[:, None, None, :])
                slot_class_indices.append(positive_label.view(1))
                slot_sample_indices.append(sample_idx)

        if not slot_tokens:
            return None

        support_tokens = torch.cat(slot_tokens, dim=0)
        support_tokens = torch.nan_to_num(support_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        class_indices = torch.cat(slot_class_indices, dim=0)
        return {
            'support_conditioned_patch_tokens': support_tokens,
            'support_branch_point_weights': support_tokens.new_ones(
                (support_tokens.shape[0], support_tokens.shape[2])
            ),
            'support_branch_class_indices': class_indices.to(
                device=support_tokens.device,
                dtype=torch.long,
            ),
            'support_branch_sample_indices': torch.tensor(
                slot_sample_indices,
                device=support_tokens.device,
                dtype=torch.long,
            ),
        }

    def _get_branch_text_features(self, global_class_indices, dtype):
        """Return projected prompt sequence, mask, and pooled text for branches."""
        text_tokens = self.text_prompt_embeddings.index_select(0, global_class_indices)
        text_mask = self.text_prompt_mask.index_select(0, global_class_indices)
        text_tokens = self.text_to_model_proj(text_tokens)
        text_tokens = F.normalize(text_tokens, dim=-1).to(dtype=dtype)
        text_weights = text_mask.to(dtype=text_tokens.dtype).unsqueeze(-1)
        text_global = (text_tokens * text_weights).sum(dim=1)
        text_global = text_global / text_weights.sum(dim=1).clamp_min(1.0)
        return text_tokens, text_mask, text_global

    def _inject_text_condition(self, branch_features, text_global):
        """Inject pooled label text through channel-wise gating."""
        text_gate = torch.sigmoid(self.text_gate_mlp(text_global)).unsqueeze(1).unsqueeze(1)
        text_bias = text_global.unsqueeze(1).unsqueeze(1)
        return (
            branch_features
            * (1.0 + self.text_inject_alpha.to(branch_features.dtype) * text_gate)
            + self.text_inject_beta.to(branch_features.dtype) * text_bias
        )


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

    def text_conditioned_pt_forward(self, x, metadata, text_tokens, text_mask):
        """Support branch forward with text injection inside each trajectory block."""
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK:
            pt_mask = metadata['pred_query_mask']
        else:
            pt_mask = metadata['pred_visibility']

        bs, temporal_dim, num_points, _ = x.shape
        x = rearrange(x, 'b t n d -> b n t d')
        pt_mask = rearrange(pt_mask, 'b t n -> b n t')
        x = rearrange(x, 'b n t d -> b (n t) d')
        pt_mask = rearrange(pt_mask, 'b n t -> b (n t)')
        if self.cfg.MODEL.USE_CLS_TOKEN:
            cls_tokens = self.cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            cls_token_mask = torch.ones(bs, 1).bool().to(x.device)
            pt_mask = torch.cat((cls_token_mask, pt_mask), dim=1)

        x = self.pos_drop(x)
        thw = [
            self.temporal_resolution,
            self.point_grid_size,
            int(num_points / self.point_grid_size),
        ]
        for _, blk in enumerate(self.blocks):
            x, _ = blk.forward_text_conditioned(
                x,
                thw,
                pt_mask,
                text_tokens,
                text_mask,
            )

        if self.cfg.MODEL.ADAPOOLING.ENABLE:
            raise NotImplementedError(
                "Text-conditioned support branch does not support adaptive pooling."
            )

        x = self.norm(x)
        if self.cfg.MODEL.USE_CLS_TOKEN:
            cls_x, patch_x = x[:, 0], x[:, 1:]
            if self.cfg.MODEL.USE_PATCH_AS_CLS:
                cls_x = patch_x.mean(dim=1)
        else:
            cls_x = x.mean(dim=1)
            patch_x = x

        cls_x = self.pre_logits(cls_x)
        patch_x = rearrange(patch_x, 'b (n t) d -> b t n d', t=temporal_dim)
        if not torch.isfinite(x).all():
            print("WARNING: nan in text-conditioned features out")
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

            else:
                sampled_feat = rearrange(feat_to_use, 'b t p q d -> b t (p q) d')
                self.point_grid_size = int(sampled_feat.shape[2] ** 0.5)
        else:
            sampled_feat = 0

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
            and 'support_mask' in metadata
            and 'episode_positive_labels' in metadata
        ):
            if self.cfg.FEW_SHOT.SOFT_LABEL_ROUTE.ENABLE:
                few_shot_aux = self._build_soft_label_routed_support_prototypes(
                    sampled_feat,
                    metadata,
                )
            else:
                few_shot_aux = self._build_support_conditioned_branches(
                    sampled_feat,
                    metadata,
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
