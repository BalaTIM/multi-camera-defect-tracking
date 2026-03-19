"""
Cameras Router
--------------
Returns metadata about registered camera nodes and their stream health.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
import time

router = APIRouter()


class CameraStatus(BaseModel):
    camera_id: str
    source: str
    status: str          # "online" | "offline" | "degraded"
    last_heartbeat: float | None
    fps: float | None


# In production this would be backed by Redis heartbeat keys.
# For now, return a stub that the edge nodes populate.
_camera_registry: dict[str, dict] = {}


@router.get("/", response_model=list[CameraStatus])
async def list_cameras():
    """List all registered cameras and their status."""
    now = time.time()
    result = []
    for cam_id, info in _camera_registry.items():
        age = now - (info.get("last_heartbeat") or 0)
        if age < 5:
            status = "online"
        elif age < 30:
            status = "degraded"
        else:
            status = "offline"

        result.append(CameraStatus(
            camera_id=cam_id,
            source=info.get("source", "unknown"),
            status=status,
            last_heartbeat=info.get("last_heartbeat"),
            fps=info.get("fps"),
        ))
    return result


@router.post("/{camera_id}/heartbeat")
async def heartbeat(camera_id: str, request: Request):
    """Edge nodes call this to register and report liveness."""
    body = await request.json()
    _camera_registry[camera_id] = {
        "source": body.get("source", ""),
        "last_heartbeat": time.time(),
        "fps": body.get("fps"),
    }
    return {"status": "ok"}
