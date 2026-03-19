"""
Pydantic Schemas
----------------
Shared request/response models used across API routes.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# ── Defects ──────────────────────────────────────────────────────────────────

class DefectBase(BaseModel):
    global_id: str
    camera_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    decision: str


class DefectResponse(DefectBase):
    id: int
    timestamp: datetime
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Objects ───────────────────────────────────────────────────────────────────

class ObjectResponse(BaseModel):
    global_id: str
    camera_id: str
    bbox: list[float]
    last_seen: float


# ── Cameras ───────────────────────────────────────────────────────────────────

class CameraHeartbeat(BaseModel):
    source: str
    fps: Optional[float] = None


class CameraStatusResponse(BaseModel):
    camera_id: str
    source: str
    status: str  # online | degraded | offline
    last_heartbeat: Optional[float] = None
    fps: Optional[float] = None


# ── Alerts ────────────────────────────────────────────────────────────────────

class AlertRequest(BaseModel):
    global_id: str
    camera_id: str
    message: str
    severity: str = Field(default="high", pattern="^(low|medium|high|critical)$")


class AlertResponse(BaseModel):
    alert_id: str
    global_id: str
    camera_id: str
    message: str
    severity: str
    triggered_at: datetime


# ── Metrics ───────────────────────────────────────────────────────────────────

class SystemMetrics(BaseModel):
    total_defects: int
    confirmed_defects: int
    uncertain_defects: int
    active_objects: int
    active_cameras: int
    uptime_s: Optional[float] = None
    messages_per_second: Optional[float] = None
    reid_latency_p99_ms: Optional[float] = None
