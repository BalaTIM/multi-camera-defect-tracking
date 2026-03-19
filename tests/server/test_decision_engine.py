"""
Unit tests for Decision Fusion Engine
"""

import pytest
import time
from server.fusion.decision_engine import DecisionEngine


class TestDecisionEngine:
    def setup_method(self):
        self.engine = DecisionEngine(temporal_window_ms=200.0, min_cameras=2)

    def test_single_camera_defect_is_uncertain(self):
        ts = time.time()
        decision = self.engine.decide("obj_1", "cam_01", defect_flag=True, timestamp=ts)
        assert decision == "UNCERTAIN"

    def test_two_camera_defect_is_confirmed(self):
        ts = time.time()
        self.engine.decide("obj_1", "cam_01", defect_flag=True, timestamp=ts)
        decision = self.engine.decide("obj_1", "cam_02", defect_flag=True, timestamp=ts + 0.05)
        assert decision == "DEFECT_CONFIRMED"

    def test_no_defect_is_ok(self):
        ts = time.time()
        self.engine.decide("obj_1", "cam_01", defect_flag=False, timestamp=ts)
        decision = self.engine.decide("obj_1", "cam_02", defect_flag=False, timestamp=ts + 0.05)
        assert decision == "OK"

    def test_stale_observations_dont_count(self):
        old_ts = time.time() - 10  # well outside window
        self.engine.decide("obj_2", "cam_01", defect_flag=True, timestamp=old_ts)
        # New observation from cam_02, fresh
        decision = self.engine.decide("obj_2", "cam_02", defect_flag=True, timestamp=time.time())
        # cam_01's old observation should be pruned, so only 1 camera → UNCERTAIN
        assert decision == "UNCERTAIN"

    def test_clear_removes_evidence(self):
        ts = time.time()
        self.engine.decide("obj_3", "cam_01", defect_flag=True, timestamp=ts)
        self.engine.clear("obj_3")
        decision = self.engine.decide("obj_3", "cam_02", defect_flag=True, timestamp=ts + 0.05)
        assert decision == "UNCERTAIN"  # only cam_02 evidence remains
