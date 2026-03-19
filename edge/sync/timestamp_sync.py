"""
Timestamp Synchronization
--------------------------
Aligns wall-clock timestamps from different edge nodes using NTP offset.
All edges run NTP; this module applies a local drift correction on top.
"""

import time
from loguru import logger


class TimestampSync:
    def __init__(self, camera_id: str, ntp_offset_ms: float = 0.0):
        """
        Args:
            camera_id: Identifier for logging.
            ntp_offset_ms: Pre-measured offset vs. server clock (milliseconds).
                           Set via config or auto-measured on startup.
        """
        self.camera_id = camera_id
        self.offset_s = ntp_offset_ms / 1000.0

    def align(self, raw_ts: float) -> float:
        """
        Apply offset correction to a raw timestamp.

        Args:
            raw_ts: Wall-clock float from time.time().

        Returns:
            Corrected timestamp in seconds (float).
        """
        return raw_ts + self.offset_s

    @staticmethod
    def now() -> float:
        """Return current corrected time."""
        return time.time()

    def measure_offset(self, server_timestamp: float) -> float:
        """
        Measure clock offset relative to server.
        Call once during startup handshake.

        Args:
            server_timestamp: Timestamp echoed back from central server.

        Returns:
            Offset in seconds (local - server).
        """
        local_now = time.time()
        self.offset_s = local_now - server_timestamp
        logger.info(
            f"[{self.camera_id}] Clock offset measured: {self.offset_s * 1000:.2f} ms"
        )
        return self.offset_s
