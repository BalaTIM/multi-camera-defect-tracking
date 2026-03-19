"""
Objects Router
--------------
Exposes globally tracked objects (cross-camera identities).
"""

from fastapi import APIRouter, Request, Query
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class ObjectResponse(BaseModel):
    global_id: str
    camera_id: str
    bbox: list[float]
    last_seen: float


@router.get("/", response_model=list[ObjectResponse])
async def list_objects(request: Request):
    """Return all currently active globally tracked objects."""
    tracker = getattr(request.app.state, "global_tracker", None)
    if tracker is None:
        return []
    return [
        ObjectResponse(
            global_id=obj.global_id,
            camera_id=obj.camera_id,
            bbox=obj.bbox,
            last_seen=obj.last_seen,
        )
        for obj in tracker.all_active()
    ]
