"""
FastAPI Application
--------------------
REST API for querying defect data, camera status, and system metrics.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes import objects, defects, cameras, metrics, alerts, stream
from utils.config import load_config
from server.storage.database import Database


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config("configs/config.yaml")
    app.state.db = Database(url=cfg["database"]["url"])
    await app.state.db.connect()
    logger.success("API ready.")
    yield
    await app.state.db.disconnect()


app = FastAPI(
    title="Multi-Camera Defect Tracking API",
    description="REST API for querying industrial defect detection results.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(objects.router, prefix="/objects", tags=["Objects"])
app.include_router(defects.router, prefix="/defects", tags=["Defects"])
app.include_router(cameras.router, prefix="/cameras", tags=["Cameras"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(stream.router, prefix="/stream", tags=["Stream"])


@app.get("/health")
async def health():
    return {"status": "ok"}
