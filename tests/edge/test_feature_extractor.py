"""
Unit tests for Feature Extractor (embedding output shape + normalization)
"""

import numpy as np
import pytest
from edge.embedding.feature_extractor import FeatureExtractor


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(embedding_dim=256)


def make_crop(h=64, w=32) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestFeatureExtractor:
    def test_single_crop_shape(self, extractor):
        crop = make_crop()
        embeddings = extractor.extract([crop])
        assert len(embeddings) == 1
        assert embeddings[0].shape == (256,)

    def test_batch_embeddings(self, extractor):
        crops = [make_crop() for _ in range(4)]
        embeddings = extractor.extract(crops)
        assert len(embeddings) == 4
        for emb in embeddings:
            assert emb.shape == (256,)

    def test_embeddings_are_l2_normalized(self, extractor):
        crops = [make_crop() for _ in range(3)]
        embeddings = extractor.extract(crops)
        for emb in embeddings:
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Embedding norm should be ~1.0, got {norm}"

    def test_empty_crops_returns_empty(self, extractor):
        result = extractor.extract([])
        assert result == []

    def test_different_crops_produce_different_embeddings(self, extractor):
        emb1 = extractor.extract([make_crop()])[0]
        emb2 = extractor.extract([make_crop()])[0]
        sim = float(np.dot(emb1, emb2))
        assert sim < 0.999, "Random crops should not produce identical embeddings"
