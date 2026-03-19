"""
API Integration Tests
----------------------
Tests the FastAPI endpoints with a running server.
Requires: uvicorn api.main:app running on localhost:8000

Run:
    pytest tests/api/test_api.py -v
"""

import pytest
import httpx

BASE = "http://localhost:8000"


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=5.0)


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestDefects:
    def test_list_defects_returns_list(self, client):
        r = client.get("/defects/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_defects_limit_param(self, client):
        r = client.get("/defects/?limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5

    def test_get_nonexistent_defect_returns_404(self, client):
        r = client.get("/defects/999999")
        assert r.status_code == 404


class TestCameras:
    def test_list_cameras(self, client):
        r = client.get("/cameras/")
        assert r.status_code == 200

    def test_camera_heartbeat(self, client):
        r = client.post(
            "/cameras/test_cam/heartbeat",
            json={"source": "rtsp://test", "fps": 25.0},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_camera_appears_after_heartbeat(self, client):
        r = client.get("/cameras/")
        cams = r.json()
        cam_ids = [c["camera_id"] for c in cams]
        assert "test_cam" in cam_ids


class TestAlerts:
    def test_trigger_alert(self, client):
        r = client.post("/alerts/", json={
            "global_id": "abc123",
            "camera_id": "cam_01",
            "message": "Test alert",
            "severity": "high",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["severity"] == "high"
        assert "alert_id" in data

    def test_list_alerts(self, client):
        r = client.get("/alerts/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestMetrics:
    def test_metrics_shape(self, client):
        r = client.get("/metrics/")
        assert r.status_code == 200
        data = r.json()
        assert "total_defects" in data
        assert "confirmed_defects" in data
        assert "active_cameras" in data


class TestStream:
    def test_stream_status(self, client):
        r = client.get("/stream/status")
        assert r.status_code == 200
        assert "active_ws_clients" in r.json()
