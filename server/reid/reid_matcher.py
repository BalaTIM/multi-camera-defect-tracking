"""
Cross-Camera Re-Identification Matcher
---------------------------------------
Maintains a global feature database and matches incoming embeddings
from different cameras to a shared global identity.

Algorithm:
  1. For each new (camera_id, local_id) pair, check if embedding matches
     any existing global ID via cosine similarity.
  2. If match found above threshold → assign same global ID.
  3. If no match → register as new global ID.
  4. Prune stale features based on TTL.
"""

import time
import uuid
from collections import defaultdict
import numpy as np
from loguru import logger


class ReIDMatcher:
    def __init__(
        self,
        threshold: float = 0.75,
        embedding_dim: int = 256,
        feature_ttl: float = 30.0,
    ):
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        self.feature_ttl = feature_ttl

        # global_id -> {"embedding": np.ndarray, "last_seen": float, "cameras": set}
        self._global_db: dict[str, dict] = {}

        # (camera_id, local_id) -> global_id  (local lookup cache)
        self._local_to_global: dict[tuple, str] = {}

    def match_or_register(
        self,
        camera_id: str,
        local_id: int,
        embedding: list[float],
        timestamp: float,
    ) -> str:
        """
        Find or create a global identity for this observation.

        Returns:
            global_id (str): UUID assigned to this object globally.
        """
        self._prune_stale(timestamp)

        # Fast lookup: same camera + local track ID seen before
        cache_key = (camera_id, local_id)
        if cache_key in self._local_to_global:
            gid = self._local_to_global[cache_key]
            self._update_entry(gid, embedding, timestamp, camera_id)
            return gid

        # Embedding similarity search
        emb = np.array(embedding, dtype=np.float32)
        best_gid, best_sim = self._find_best_match(emb)

        if best_sim >= self.threshold:
            gid = best_gid
            logger.debug(f"Re-ID match: ({camera_id}, {local_id}) → {gid} (sim={best_sim:.3f})")
        else:
            gid = str(uuid.uuid4())[:8]  # short unique ID
            logger.info(f"New global ID registered: {gid} from {camera_id}")

        self._global_db[gid] = {
            "embedding": emb,
            "last_seen": timestamp,
            "cameras": {camera_id},
        }
        self._local_to_global[cache_key] = gid
        return gid

    def _find_best_match(self, emb: np.ndarray) -> tuple[str | None, float]:
        best_gid = None
        best_sim = -1.0
        for gid, entry in self._global_db.items():
            sim = float(np.dot(emb, entry["embedding"]))  # both L2-normalized
            if sim > best_sim:
                best_sim = sim
                best_gid = gid
        return best_gid, best_sim

    def _update_entry(self, gid: str, embedding: list[float], timestamp: float, camera_id: str):
        """EMA update of stored embedding for stability."""
        if gid not in self._global_db:
            return
        alpha = 0.1  # smoothing factor
        old = self._global_db[gid]["embedding"]
        new = np.array(embedding, dtype=np.float32)
        updated = (1 - alpha) * old + alpha * new
        # Re-normalize
        norm = np.linalg.norm(updated)
        self._global_db[gid]["embedding"] = updated / norm if norm > 0 else updated
        self._global_db[gid]["last_seen"] = timestamp
        self._global_db[gid]["cameras"].add(camera_id)

    def _prune_stale(self, current_ts: float):
        stale = [
            gid for gid, entry in self._global_db.items()
            if current_ts - entry["last_seen"] > self.feature_ttl
        ]
        for gid in stale:
            del self._global_db[gid]
        # Clean local cache
        self._local_to_global = {
            k: v for k, v in self._local_to_global.items()
            if v in self._global_db
        }

    @property
    def active_ids(self) -> list[str]:
        return list(self._global_db.keys())
