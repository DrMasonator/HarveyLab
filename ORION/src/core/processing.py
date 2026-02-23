"""Image processing and exposure control."""

from __future__ import annotations

import logging

import numpy as np

from ORION.config import Config
from ORION.src.drivers.hardware import LaserSystem

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process_image(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply Bayer slicing and return (processed_img, virtual_pixel_size_um)."""
        mode = self.config.BAYER_MODE.upper()
        pixel_um = float(self.config.PIXEL_SIZE_UM)

        if mode == "RAW":
            return img, pixel_um
        if mode == "RED":
            return img[1::2, 0::2], pixel_um * 2
        if mode == "GREEN":
            g1 = img[0::2, 0::2].astype(np.uint16)
            g2 = img[1::2, 1::2].astype(np.uint16)
            return ((g1 + g2) // 2).astype(np.uint8), pixel_um
        if mode == "BLUE":
            return img[0::2, 1::2], pixel_um * 2

        logger.warning("Unknown BAYER_MODE '%s'. Falling back to RAW.", mode)
        return img, pixel_um


class ExposureController:
    def __init__(self, config: Config, system: LaserSystem):
        self.config = config
        self.system = system

    def handle_auto_exposure(self, max_val: float) -> bool:
        """Adjust exposure to keep peak intensity in target range."""
        current_exp = float(self.system.current_exposure)
        new_exp = current_exp
        target_center = (self.config.TARGET_BRIGHTNESS_MIN + self.config.TARGET_BRIGHTNESS_MAX) / 2.0

        if max_val >= self.config.ABSOLUTE_SATURATION:
            new_exp = current_exp * 0.5
        elif max_val < self.config.LOW_SIGNAL_THRESHOLD:
            new_exp = current_exp * 1.5
        elif max_val > self.config.TARGET_BRIGHTNESS_MAX:
            ratio = target_center / float(max_val)
            new_exp = current_exp * max(0.8, ratio)
        elif max_val < self.config.TARGET_BRIGHTNESS_MIN:
            ratio = target_center / float(max_val)
            new_exp = current_exp * min(1.2, ratio)

        new_exp = max(self.config.MIN_EXPOSURE_MS, min(self.config.MAX_EXPOSURE_MS, new_exp))

        if abs(new_exp - current_exp) / current_exp > 0.05:
            self.system.set_exposure(new_exp)
            return True
        return False
