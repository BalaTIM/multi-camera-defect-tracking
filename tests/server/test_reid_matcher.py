"""
Unit tests for ReID Matcher
"""

import numpy as np
import pytest
import time
from server.reid.reid_matcher import ReIDMatcher


def make_emb(seed: int, dim: int = 256) -> list[float]:
    """Create a normalized random embedding."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


class TestReIDMatcher:
    def setup_method(self):
        self.reid = ReIDMatcher(threshold=0.75, embedding_dim=256, feature_ttl=30.0)

    def test_new_object_gets_unique_id(self):
        gid = self.reid.match_or_register("cam_01", 1, make_emb(42), time.time())
        assert isinstance(gid, str)
        assert len(gid) > 0

    def test_same_local_track_returns_same_global_id(self):
        ts = time.time()
        gid1 = self.reid.match_or_register("cam_01", 1, make_emb(42), ts)
        gid2 = self.reid.match_or_register("cam_01", 1, make_emb(42), ts + 0.1)
        assert gid1 == gid2

    def test_similar_embeddings_match_across_cameras(self):
        emb = make_emb(99)
        # Slightly perturbed version of same embedding (simulates same object, different view)
        noise = np.array(make_emb(99)) * 0.98 + np.random.default_rng(0).random(256) * 0.02
        noise = (noise / np.linalg.norm(noise)).tolist()

        ts = time.time()
        gid1 = self.reid.match_or_register("cam_01", 1, emb, ts)
        gid2 = self.reid.match_or_register("cam_02", 1, noise, ts + 0.05)
        assert gid1 == gid2, "Similar embeddings from different cameras should share global ID"

    def test_different_objects_get_different_ids(self):
        ts = time.time()
        gid1 = self.reid.match_or_register("cam_01", 1, make_emb(1), ts)
        gid2 = self.reid.match_or_register("cam_01", 2, make_emb(100), ts)
        assert gid1 != gid2

    def test_stale_entries_are_pruned(self):
        ts = time.time() - 100  # old timestamp
        self.reid.match_or_register("cam_01", 1, make_emb(1), ts)
        # Trigger pruning with current time
        self.reid._prune_stale(time.time())
        assert len(self.reid.active_ids) == 0
