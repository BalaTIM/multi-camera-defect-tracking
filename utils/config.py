"""
Config Loader
-------------
Loads YAML config file and returns as a nested dict.
"""

import yaml
from pathlib import Path
from loguru import logger


def load_config(path: str = "configs/config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg
