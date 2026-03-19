"""
Logging Setup
-------------
Configures loguru with rotating file output and structured format.
"""

import sys
from loguru import logger
from pathlib import Path


def setup_logger(camera_id: str = "edge", log_dir: str = "logs", level: str = "INFO"):
    """
    Configure loguru for edge node.

    Args:
        camera_id: Included in every log line for easy grep.
        log_dir:   Directory where rotating log files are written.
        level:     Minimum log level (DEBUG / INFO / WARNING / ERROR).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<cyan>{camera_id}</cyan> | "
        "<level>{level: <8}</level> | "
        "<white>{message}</white>"
    )

    # Console
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    # Rotating file (10 MB, keep 7 days)
    logger.add(
        log_path / f"{camera_id}.log",
        format=fmt,
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    return logger
