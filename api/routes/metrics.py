"""
Metrics Router
--------------
Exposes system performance and defect statistics.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class SystemMetrics(BaseModel):
    total_defects: int
    confirmed_defects: int
    uncertain_defects: int
    active_objects: int
    active_cameras: int


@router.get("/", response_model=SystemMetrics)
async def get_metrics(request: Request):
    """Return aggregated system performance metrics."""
    db = request.app.state.db
    records = await db.get_recent_defects(limit=10000)

    confirmed = sum(1 for r in records if r.decision == "DEFECT_CONFIRMED")
    uncertain = sum(1 for r in records if r.decision == "UNCERTAIN")

    tracker = getattr(request.app.state, "global_tracker", None)
    active_objects = len(tracker.all_active()) if tracker else 0

    from api.routes.cameras import _camera_registry
    import time
    active_cameras = sum(
        1 for info in _camera_registry.values()
        if time.time() - (info.get("last_heartbeat") or 0) < 10
    )

    return SystemMetrics(
        total_defects=len(records),
        confirmed_defects=confirmed,
        uncertain_defects=uncertain,
        active_objects=active_objects,
        active_cameras=active_cameras,
    )
