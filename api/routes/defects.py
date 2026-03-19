"""
Defects Router
--------------
"""

from fastapi import APIRouter, Request, Query
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class DefectResponse(BaseModel):
    id: int
    global_id: str
    camera_id: str
    label: str
    confidence: float
    bbox: list[float]
    decision: str
    timestamp: datetime

    class Config:
        from_attributes = True


@router.get("/", response_model=list[DefectResponse])
async def list_defects(
    request: Request,
    limit: int = Query(default=50, le=500),
    decision: str | None = Query(default=None),
):
    """Return recent defect events."""
    records = await request.app.state.db.get_recent_defects(limit=limit)
    result = []
    for r in records:
        if decision and r.decision != decision:
            continue
        result.append(DefectResponse(
            id=r.id,
            global_id=r.global_id,
            camera_id=r.camera_id,
            label=r.label,
            confidence=r.confidence,
            bbox=[r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2],
            decision=r.decision,
            timestamp=r.timestamp,
        ))
    return result


@router.get("/{defect_id}", response_model=DefectResponse)
async def get_defect(defect_id: int, request: Request):
    """Get a single defect record by ID."""
    from sqlalchemy import select
    from server.storage.database import DefectRecord
    async with request.app.state.db.session_factory() as session:
        result = await session.get(DefectRecord, defect_id)
        if not result:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Defect not found")
        return DefectResponse(
            id=result.id,
            global_id=result.global_id,
            camera_id=result.camera_id,
            label=result.label,
            confidence=result.confidence,
            bbox=[result.bbox_x1, result.bbox_y1, result.bbox_x2, result.bbox_y2],
            decision=result.decision,
            timestamp=result.timestamp,
        )
