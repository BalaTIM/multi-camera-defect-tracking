"""
Camera Stream Capture
---------------------
Wraps OpenCV VideoCapture with reconnection logic and frame rate control.
Supports RTSP, local files, and webcam indices.
"""

import time
import cv2
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class CameraStream:
    def __init__(self, source: str | int, fps: int = 25, reconnect_delay: float = 2.0):
        """
        Args:
            source: RTSP URL string, file path, or integer webcam index.
            fps: Target frames per second (throttles capture if source is faster).
            reconnect_delay: Seconds to wait before reconnecting on failure.
        """
        self.source = source
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ConnectionError(f"Cannot open video source: {self.source}")
        logger.info(f"Camera source opened: {self.source}")
        return cap

    def read(self):
        """
        Generator that yields (frame: np.ndarray, timestamp: float) tuples.
        Automatically reconnects on failure.
        """
        self._cap = self._open()
        last_frame_time = 0.0

        while True:
            now = time.monotonic()
            elapsed = now - last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"Stream lost ({self.source}). Reconnecting...")
                self._cap.release()
                time.sleep(self.reconnect_delay)
                try:
                    self._cap = self._open()
                except Exception as e:
                    logger.error(f"Reconnect failed: {e}")
                    break
                continue

            last_frame_time = time.monotonic()
            yield frame, time.time()  # wall-clock timestamp for sync

    def release(self):
        if self._cap:
            self._cap.release()
            logger.info("Camera stream released.")
