from dataclasses import dataclass

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
    
    LOW_SIGNAL_THRESHOLD: int = 50
    
    # Options: 'RAW', 'RED', 'GREEN', 'BLUE'
    BAYER_MODE: str = 'RED'
    
    NOISE_CUTOFF_PERCENT: float = 0.15
    MEASURE_AVERAGE_COUNT: int = 20
    
    
    MIN_Z_MM: float = 0.0
    MAX_Z_MM: float = 13.0
    BACKLASH_DIST_MM: float = 0.005
    
    SEARCH_WINDOW_MM: float = 0.3
    SEARCH_TOLERANCE: float = 0.0005
