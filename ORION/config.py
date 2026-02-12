from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional
import json

@dataclass
class Config:
    MOTOR_SERIAL: str = "27263086"
    Z812_SCALE: int = 34304
    PIXEL_SIZE_UM: float = 1.4
    WAVELENGTH_NM: float = 625.0
    
    
    START_EXPOSURE_MS: float = 1.0
    MIN_EXPOSURE_MS: float = 0.017
    MAX_EXPOSURE_MS: float = 1144.0
    
    TARGET_BRIGHTNESS_MIN: int = 178  # ~70%
    TARGET_BRIGHTNESS_MAX: int = 230  # ~90%
    ABSOLUTE_SATURATION: int = 254
    
    LOW_SIGNAL_THRESHOLD: int = 50

    # Options: 'RAW', 'RED', 'GREEN', 'BLUE'
    BAYER_MODE: str = 'RED'
    
    NOISE_CUTOFF_PERCENT: float = 0.15
    MEASURE_AVERAGE_COUNT: int = 20
    FIND_BEAM_AVERAGE_COUNT: int = 5
    FIND_BEAM_SKIP_AE: bool = True
    
    
    MIN_Z_MM: float = 0.0
    MAX_Z_MM: float = 13.0
    BACKLASH_DIST_MM: float = 0.005
    
    SEARCH_WINDOW_MM: float = 0.3
    SEARCH_TOLERANCE: float = 0.0005

    # ROI tracking (processed-image coordinates)
    ROI_MIN_SIZE_PX: int = 400
    ROI_MAX_SIZE_PX: int = 2600
    ROI_SEARCH_BLOCK: int = 16
    ROI_EXPAND_THRESHOLD: float = 0.65  # Expand if 3*D4s > threshold * ROI
    ROI_SHRINK_THRESHOLD: float = 0.15  # Shrink if 3*D4s < threshold * ROI
    ROI_ADAPT_HYSTERESIS_FRAMES: int = 3
    ROI_EDGE_PEAK_FRACTION: float = 0.03  # Expand if ROI border has >3% of peak

    @staticmethod
    def default_path() -> Path:
        return Path.home() / ".orion_config.json"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        cfg = cls()
        cfg_path = path or cls.default_path()
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text())
                for f in fields(cfg):
                    if f.name not in data:
                        continue
                    raw = data[f.name]
                    # Basic type coercion
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
                # If config load fails, fall back to defaults
                pass
        return cfg

    def save(self, path: Optional[Path] = None) -> None:
        cfg_path = path or self.default_path()
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        cfg_path.write_text(json.dumps(data, indent=2, sort_keys=True))
