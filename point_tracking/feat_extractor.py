import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from einops import rearrange
from clustering import TorchKMeansVectorizedCluster

VALID_MODEL_TYPES = {"dino", "clip_vit_b16"}

class feature_extract(nn.Module):
    def __init__(self, model_type="dino"):
        super().__init__()
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model type: {model_type}")

        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        self.dinov2 = None
        self.clip_model = None
        self.dino_transform = None
        self.clip_transform = None

        self._load_model(model_type)

    def _load_model(self, model_type):
        if model_type == "dino":
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                trust_repo=True,
                skip_validation=True,
            )
            self.dinov2.to(self.device)
            self.dinov2.eval()
            self.dino_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        elif model_type == "clip_vit_b16":
            import clip
            self.clip_model, self.clip_transform = clip.load(
                "ViT-B/16",
                device=self.device,
                jit=False,
            )
            self.clip_model.eval()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def _process_frames(self, frames, transform):
        processed_frames = []
        batch_size, num_frames = frames.shape[:2]
        for i in range(batch_size):
            for j in range(num_frames):
                frame_pil = Image.fromarray(frames[i, j])
                frame_tensor = transform(frame_pil)
                processed_frames.append(frame_tensor)
        return torch.stack(processed_frames)

    def _get_dino_features(self, frames):
        x = self._process_frames(frames, self.dino_transform).to(self.device)
        return self.dinov2.forward_features(x)['x_norm_patchtokens']

    def _get_clip_patch_tokens(self, frames):
        x = self._process_frames(frames, self.clip_transform).to(self.device)
        visual = self.clip_model.visual
        x = x.type(visual.conv1.weight.dtype)
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
        return x[:, 1:, :]


    @torch.no_grad()
    def forward(self, frames, model_type=None):
        """
        Args:
            frames: numpy array of shape (bs, num_frames, height, width, channel).
            model_type: one of 'dino' or 'clip_vit_b16'
        """
        batch_size, num_frames, _, _, _ = frames.shape
        model_type = model_type or self.model_type

        if model_type != self.model_type:
            raise ValueError(
                f"feature_extract initialized for {self.model_type}, got {model_type}"
            )

        if model_type == 'dino':
            feat = self._get_dino_features(frames)
        elif model_type == "clip_vit_b16":
            feat = self._get_clip_patch_tokens(frames)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Reshape features
        feat = feat.float()
        feat = rearrange(feat, '(b t) p d -> b t p d', b=batch_size, t=num_frames)
        patch_size = int(feat.shape[2] ** 0.5)
        feat = rearrange(feat, 'b t (p q) d -> b t p q d', p=patch_size)

        return feat

    def cluster_features(self, feat, method='dbscan', n_clusters=8,
                         prev_centers=None, global_clustering=False, use_torch=False):
        """
        Args:
            feat: Features to cluster
            method: Clustering method ('dbscan' or 'kmeans')
            n_clusters: Number of clusters for kmeans
            prev_centers: Previous cluster centers for temporal consistency
            global_clustering: If True, cluster all frames together
        """
        if global_clustering:
            # For global clustering, reshape to (n_frames * n_patches, n_features)
            n_frames, n_patches_h, n_patches_w, feat_dim = feat.shape

            if use_torch:
                feat_2d = feat.reshape(-1, feat_dim)
            else:
                feat_2d = feat.reshape(-1, feat_dim).cpu().numpy()

        else:
            # Original per-frame clustering
            n_patches = feat.shape[0] * feat.shape[1]

            if use_torch:
                feat_2d = feat.reshape(n_patches, -1)
            else:
                feat_2d = feat.reshape(n_patches, -1).cpu().numpy()

        if method == 'dbscan':
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(feat_2d)
            labels = clustering.labels_
            centers = None
        elif method == 'kmeans':
            if use_torch:
                #print('verbose...usetorchkmeans')
                device = feat_2d.device
                clustering = TorchKMeansVectorizedCluster(n_clusters=n_clusters)
                if prev_centers is not None:
                    prev_centers = torch.tensor(prev_centers).to(device)
                labels, centers = clustering(feat_2d, prev_centers=prev_centers)
                labels = labels.cpu().numpy()
                centers = centers.cpu().numpy()

            else:
                from sklearn.cluster import KMeans
                if prev_centers is not None:
                    clustering = KMeans(n_clusters=n_clusters, random_state=42,
                                            init=prev_centers, n_init=1)
                else:
                    clustering = KMeans(n_clusters=n_clusters, random_state=42)

                clustering.fit(feat_2d)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
        else:
            raise ValueError("Method must be either 'dbscan' or 'kmeans'")

        if global_clustering:
            # Reshape labels back to (n_frames, patch_h, patch_w)
            labels = labels.reshape(n_frames, n_patches_h, n_patches_w)
        else:
            # Reshape labels back to patch grid
            labels = labels.reshape(feat.shape[0], feat.shape[1])

        return labels, centers
