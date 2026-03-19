"""
Feature Extractor
-----------------
Extracts normalized appearance embeddings from cropped object images.
Used for cross-camera re-identification (Re-ID).

Backbone: ResNet50 pretrained → custom projection head → L2-normalized vector
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from loguru import logger


class EmbeddingHead(nn.Module):
    """Projection head: reduces ResNet output to target embedding dim."""

    def __init__(self, in_features: int = 2048, embedding_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FeatureExtractor:
    def __init__(
        self,
        embedding_dim: int = 256,
        weights_path: str | None = None,
        device: str | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop classifier
        self.backbone.eval()

        # Projection head
        self.head = EmbeddingHead(in_features=2048, embedding_dim=embedding_dim)
        self.head.eval()

        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.head.load_state_dict(state)
            logger.info(f"Loaded Re-ID weights from {weights_path}")
        else:
            logger.warning("No Re-ID weights provided — using ImageNet features (demo mode).")

        self.backbone.to(self.device)
        self.head.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """
        Extract L2-normalized embeddings for a batch of BGR image crops.

        Args:
            crops: List of np.ndarray (H x W x 3, BGR).

        Returns:
            List of 1-D float32 numpy arrays of shape (embedding_dim,).
        """
        if not crops:
            return []

        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                tensors.append(torch.zeros(3, 128, 64))
                continue
            rgb = crop[:, :, ::-1].copy()   # BGR → RGB
            tensors.append(self.transform(rgb))

        batch = torch.stack(tensors).to(self.device)

        features = self.backbone(batch)                  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        embeddings = self.head(features)                 # (B, embedding_dim)

        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return [e.cpu().numpy() for e in embeddings]
