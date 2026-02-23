"""Application configuration with simple JSON persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Hardware
    MOTOR_SERIAL: str = "27263086"
    Z812_SCALE: int = 34304
    PIXEL_SIZE_UM: float = 1.4
    WAVELENGTH_NM: float = 625.0

    # Exposure control
    START_EXPOSURE_MS: float = 1.0
    MIN_EXPOSURE_MS: float = 0.017
    MAX_EXPOSURE_MS: float = 1144.0
    TARGET_BRIGHTNESS_MIN: int = 178
    TARGET_BRIGHTNESS_MAX: int = 230
    ABSOLUTE_SATURATION: int = 254
    LOW_SIGNAL_THRESHOLD: int = 50

    # Image processing
    BAYER_MODE: str = "RED"  # RAW, RED, GREEN, BLUE

    # Beam measurement
    NOISE_CUTOFF_PERCENT: float = 0.15
    MEASURE_AVERAGE_COUNT: int = 20
    FIND_BEAM_AVERAGE_COUNT: int = 5
    FIND_BEAM_SKIP_AE: bool = True

    # Motion
    MIN_Z_MM: float = 0.0
    MAX_Z_MM: float = 13.0
    BACKLASH_DIST_MM: float = 0.005
    SEARCH_WINDOW_MM: float = 0.3
    SEARCH_TOLERANCE: float = 0.0005

    # ROI tracking (processed-image coordinates)
    ROI_MIN_SIZE_PX: int = 400
    ROI_MAX_SIZE_PX: int = 2600
    ROI_SEARCH_BLOCK: int = 16
    ROI_EXPAND_THRESHOLD: float = 0.65
    ROI_SHRINK_THRESHOLD: float = 0.15
    ROI_ADAPT_HYSTERESIS_FRAMES: int = 3
    ROI_EDGE_PEAK_FRACTION: float = 0.03

    @staticmethod
    def default_path() -> Path:
        return Path.home() / ".orion_config.json"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        cfg = cls()
        cfg_path = path or cls.default_path()
        if not cfg_path.exists():
            return cfg

        try:
            data = json.loads(cfg_path.read_text())
        except Exception:
            logger.exception("Failed to read config file: %s", cfg_path)
            return cfg

        for f in fields(cfg):
            if f.name not in data:
                continue
            raw = data[f.name]
            try:
                if f.type is bool:
                    val = bool(raw)
                elif f.type is int:
                    val = int(raw)
                elif f.type is float:
                    val = float(raw)
                else:
                    val = raw
                setattr(cfg, f.name, val)
            except Exception:
                logger.warning("Ignoring invalid config value for %s", f.name)

        cfg.normalize()
        return cfg

    def save(self, path: Optional[Path] = None) -> None:
        cfg_path = path or self.default_path()
        cfg_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    def normalize(self) -> None:
        if self.MIN_EXPOSURE_MS > self.MAX_EXPOSURE_MS:
            self.MIN_EXPOSURE_MS, self.MAX_EXPOSURE_MS = self.MAX_EXPOSURE_MS, self.MIN_EXPOSURE_MS
        if self.TARGET_BRIGHTNESS_MIN > self.TARGET_BRIGHTNESS_MAX:
            self.TARGET_BRIGHTNESS_MIN, self.TARGET_BRIGHTNESS_MAX = (
                self.TARGET_BRIGHTNESS_MAX,
                self.TARGET_BRIGHTNESS_MIN,
            )
        if self.ROI_MIN_SIZE_PX > self.ROI_MAX_SIZE_PX:
            self.ROI_MIN_SIZE_PX, self.ROI_MAX_SIZE_PX = self.ROI_MAX_SIZE_PX, self.ROI_MIN_SIZE_PX
