# Architecture Deep Dive

## Overview

This document explains the design decisions behind each major component.

---

## Edge Node

Each camera runs an independent edge process that is fully self-contained.
If the central server becomes unreachable, the edge node continues detecting
and buffering messages locally (Kafka provides durable message retention).

**Key pipeline:**
```
Frame → Resize → YOLOv8 → Crop detections → ResNet50 embedding → DeepSORT → Kafka
```

**Why DeepSORT at the edge?**
Without local tracking, every frame would emit a fresh "new object" event even
for the same physical part. DeepSORT provides stable local IDs so that the
Kafka message rate is bounded by number of tracked objects, not raw detections.

---

## Streaming Layer (Kafka)

Kafka topics are partitioned by `camera_id` so that all messages from the
same camera arrive in order. The central server can scale horizontally by
adding consumer group members — each consumer handles a subset of partitions.

**Message retention:** 1 hour by default. Edge nodes can recover from server
downtime by replaying messages.

---

## Cross-Camera Re-ID

The Re-ID module maintains a global feature database:

```
global_db = {
  "a3f1b2": {
    "embedding": np.ndarray(256,),   # L2-normalized, EMA-updated
    "last_seen": float,              # epoch seconds
    "cameras": {"cam_01", "cam_03"}, # cameras that have seen this object
  },
  ...
}
```

**Matching algorithm:**
1. Incoming embedding is compared against all entries via dot product
   (equivalent to cosine similarity since vectors are L2-normalized).
2. If best similarity ≥ threshold → reuse global ID.
3. Otherwise → assign new UUID.
4. EMA update (α=0.1) prevents embedding drift while remaining robust to
   viewpoint changes.

**Threshold calibration:**
- Too low → false positives (different objects merged)
- Too high → false negatives (same object gets multiple IDs)
- Default 0.75 works well for same-object across viewpoints; tune on your data.

---

## Decision Fusion Engine

Temporal window fusion prevents false alerts:
- A single camera seeing a defect → `UNCERTAIN`
- Two or more cameras seeing the same global object defect within the
  temporal window → `DEFECT_CONFIRMED`

This reduces false positives significantly in noisy industrial environments.

---

## Database Schema

```sql
CREATE TABLE defects (
    id          SERIAL PRIMARY KEY,
    global_id   VARCHAR(16)  NOT NULL,
    camera_id   VARCHAR(32)  NOT NULL,
    label       VARCHAR(64)  NOT NULL,
    confidence  FLOAT        NOT NULL,
    bbox_x1     FLOAT, bbox_y1 FLOAT, bbox_x2 FLOAT, bbox_y2 FLOAT,
    decision    VARCHAR(32)  NOT NULL,  -- DEFECT_CONFIRMED | UNCERTAIN
    timestamp   TIMESTAMP    NOT NULL,
    created_at  TIMESTAMP    DEFAULT NOW()
);
CREATE INDEX idx_defects_global_id ON defects (global_id);
CREATE INDEX idx_defects_timestamp ON defects (timestamp DESC);
```

---

## Fault Tolerance

| Failure | Behaviour |
|---------|-----------|
| Central server down | Edge nodes keep publishing to Kafka (buffered) |
| Single camera fails | Other cameras continue; Re-ID gracefully handles absence |
| Kafka broker restart | Consumer reconnects with exponential backoff (tenacity) |
| DB connection lost | FastAPI returns 503; server retries on next message |

---

## Scaling

- **Horizontal edge scaling:** Add more cameras by launching new edge
  containers. No server changes needed.
- **Horizontal server scaling:** Add Kafka consumer group members.
  Re-ID state is currently in-memory — for multi-instance servers, migrate
  the feature DB to Redis.
- **Model optimization:** Export YOLOv8 to TensorRT INT8 for 3–4× speedup
  on Jetson devices.
