"""
Alerts Router
-------------
Allows external systems to query or manually trigger defect alerts.
In production, hook into PagerDuty / Slack / email.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from datetime import datetime
import time

router = APIRouter()


class AlertRequest(BaseModel):
    global_id: str
    camera_id: str
    message: str
    severity: str = "high"  # low | medium | high | critical


class AlertResponse(BaseModel):
    alert_id: str
    global_id: str
    camera_id: str
    message: str
    severity: str
    triggered_at: datetime


_alert_log: list[AlertResponse] = []


@router.post("/", response_model=AlertResponse)
async def trigger_alert(payload: AlertRequest):
    """Manually trigger a defect alert."""
    import uuid
    alert = AlertResponse(
        alert_id=str(uuid.uuid4())[:8],
        global_id=payload.global_id,
        camera_id=payload.camera_id,
        message=payload.message,
        severity=payload.severity,
        triggered_at=datetime.utcnow(),
    )
    _alert_log.append(alert)
    # TODO: send to Slack / PagerDuty / webhook
    return alert


@router.get("/", response_model=list[AlertResponse])
async def list_alerts(limit: int = 50):
    """List recent alerts."""
    return _alert_log[-limit:]
