import numpy as np
from ORION.config import Config
from ORION.src.drivers.hardware import LaserSystem

class ImageProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process_image(self, img: np.ndarray) -> tuple:
        """
        Applies Bayer mask slicing.
        Returns (processed_img, virtual_pixel_size)
        """
        mode = self.config.BAYER_MODE
        virtual_pixel_size = self.config.PIXEL_SIZE_UM
        
        if mode == 'RAW':
            return img, virtual_pixel_size
        elif mode == 'RED':
            # GBRG pattern: Red is (1,0)
            virtual_pixel_size = self.config.PIXEL_SIZE_UM * 2
            return img[1::2, 0::2], virtual_pixel_size
        elif mode == 'GREEN':
            # GBRG pattern: Green is (0,0) and (1,1)
            g1 = img[0::2, 0::2].astype(np.uint16)
            g2 = img[1::2, 1::2].astype(np.uint16)
            return ((g1 + g2) // 2).astype(np.uint8), virtual_pixel_size
        elif mode == 'BLUE':
            # GBRG pattern: Blue is (0,1)
            virtual_pixel_size = self.config.PIXEL_SIZE_UM * 2
            return img[0::2, 1::2], virtual_pixel_size
        return img, virtual_pixel_size

class ExposureController:
    def __init__(self, config: Config, system: LaserSystem):
        self.config = config
        self.system = system

    def handle_auto_exposure(self, max_val: float) -> bool:
        current_exp = self.system.current_exposure
        new_exp = current_exp
        
        target_center = (self.config.TARGET_BRIGHTNESS_MIN + self.config.TARGET_BRIGHTNESS_MAX) / 2.0
        
        if max_val >= self.config.ABSOLUTE_SATURATION:
            new_exp = current_exp * 0.5
        elif max_val < self.config.LOW_SIGNAL_THRESHOLD:
            new_exp = current_exp * 1.5
        elif max_val > self.config.TARGET_BRIGHTNESS_MAX:
            ratio = target_center / float(max_val)
            ratio = max(0.8, ratio) 
            new_exp = current_exp * ratio
        elif max_val < self.config.TARGET_BRIGHTNESS_MIN:
             ratio = target_center / float(max_val)
             ratio = min(1.2, ratio)
             new_exp = current_exp * ratio

        new_exp = max(self.config.MIN_EXPOSURE_MS, min(self.config.MAX_EXPOSURE_MS, new_exp))
        
        if abs(new_exp - current_exp) / current_exp > 0.05:
            self.system.set_exposure(new_exp)
            return True 
        return False
