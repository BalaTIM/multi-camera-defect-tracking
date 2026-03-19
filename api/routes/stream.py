"""
WebSocket Stream Router
-----------------------
Pushes real-time defect events to connected dashboard clients.
Clients connect to ws://host:8000/stream/ws and receive JSON events.
"""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._active.append(ws)
        logger.info(f"WebSocket connected. Total clients: {len(self._active)}")

    def disconnect(self, ws: WebSocket):
        self._active.remove(ws)
        logger.info(f"WebSocket disconnected. Total clients: {len(self._active)}")

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected clients."""
        payload = json.dumps(message)
        dead = []
        for ws in self._active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.remove(ws)

    @property
    def client_count(self) -> int:
        return len(self._active)


# Module-level manager — shared across requests
manager = ConnectionManager()


@router.websocket("/ws")
async def stream_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time defect event streaming.

    Message format:
    {
      "type": "defect" | "heartbeat",
      "global_id": "a3f1b2",
      "camera_id": "cam_01",
      "decision": "DEFECT_CONFIRMED",
      "label": "crack",
      "confidence": 0.94,
      "bbox": [x1, y1, x2, y2],
      "timestamp": 1712345678.0
    }
    """
    await manager.connect(websocket)
    try:
        # Send welcome + heartbeat loop
        await websocket.send_json({"type": "connected", "clients": manager.client_count})
        while True:
            # Keep connection alive with periodic heartbeat
            await asyncio.sleep(5)
            await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.get("/status")
async def stream_status():
    """Return current stream connection status."""
    return {
        "active_ws_clients": manager.client_count,
        "status": "running",
    }
