import sys
import time
import ctypes
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from ORION.config import Config

class LaserSystem(ABC):
    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None: pass
    @abstractmethod
    def get_raw_image(self) -> Optional[np.ndarray]: pass
    @abstractmethod
    def move_motor_precise(self, target_mm: float) -> None: pass
    @abstractmethod
    def close(self) -> None: pass
    @property
    @abstractmethod
    def current_exposure(self) -> float: pass
    @abstractmethod
    def set_soft_limit(self, limit_mm: float) -> None: pass
    @property
    @abstractmethod
    def soft_limit(self) -> float: pass
    @property
    @abstractmethod
    def current_position(self) -> float: pass

class RealLaserSystem(LaserSystem):
    def __init__(self, config: Config = Config()):
        print("--- Initializing Hardware ---")
        self.config = config
        self._current_exposure = self.config.START_EXPOSURE_MS
        self._current_pos = 0.0
        
        from pylablib.devices import Thorlabs
        from ids_peak import ids_peak
        
        try:
            self.motor = Thorlabs.KinesisMotor(self.config.MOTOR_SERIAL, scale=self.config.Z812_SCALE)
            print(f"[OK] Motor {self.config.MOTOR_SERIAL} connected.")
            print("Homing Motor...")
            self.motor.home()
            self.motor.wait_for_home()
            self._current_pos = 0.0
            self._soft_limit = self.config.MAX_Z_MM
        except Exception as e:
            sys.exit(1)

        # Camera
        try:
            ids_peak.Library.Initialize()
            dev_mgr = ids_peak.DeviceManager.Instance()
            dev_mgr.Update()
            if dev_mgr.Devices().empty(): raise Exception("No Camera")
            
            self.device = dev_mgr.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            self.node_map = self.device.RemoteDevice().NodeMaps()[0]
            self.stream = self.device.DataStreams()[0].OpenDataStream()
            
            self.set_exposure(self.config.START_EXPOSURE_MS)

            payload = self.node_map.FindNode("PayloadSize").Value()
            for _ in range(3):
                buf = self.stream.AllocAndAnnounceBuffer(payload)
                self.stream.QueueBuffer(buf)
                
            self.stream.StartAcquisition()
            self.node_map.FindNode("AcquisitionStart").Execute()
            self.node_map.FindNode("TLParamsLocked").SetValue(1)
            print("[OK] Camera Stream Started")
        except Exception as e:
            print(f"[FAIL] Camera error: {e}")
            sys.exit(1)

    @property
    def current_exposure(self) -> float: return self._current_exposure
    
    @property
    def current_position(self) -> float: return self.motor.get_position()
    
    @property
    def soft_limit(self) -> float: return self._soft_limit

    def set_soft_limit(self, limit_mm: float) -> None:
        self._soft_limit = limit_mm
        print(f"[Hardware] Soft limit set to {limit_mm:.3f} mm")

    def set_exposure(self, exposure_ms: float) -> None:
        try:
            exposure_us = exposure_ms * 1000.0
            self.node_map.FindNode("ExposureTime").SetValue(exposure_us)
            self._current_exposure = exposure_ms
        except Exception: pass

    def get_raw_image(self) -> Optional[np.ndarray]:
        buf = None
        try:
            buf = self.stream.WaitForFinishedBuffer(5000)
            w, h, size = int(buf.Width()), int(buf.Height()), int(buf.Size())
            ptr_int = int(buf.BasePtr())
            c_data = (ctypes.c_uint8 * size).from_address(ptr_int)
            raw_img = np.frombuffer(c_data, dtype=np.uint8).reshape((h, w))
            return raw_img.copy()
        except Exception:
            return None
        finally:
            if buf: self.stream.QueueBuffer(buf)

    def move_motor_precise(self, target_mm: float) -> None:
        if target_mm < self.config.MIN_Z_MM: target_mm = self.config.MIN_Z_MM
        if target_mm > self._soft_limit: 
            print(f"[Safety] Target {target_mm:.3f} exceeds soft limit {self._soft_limit:.3f}. Clamping.")
            target_mm = self._soft_limit
        if target_mm > self.config.MAX_Z_MM: target_mm = self.config.MAX_Z_MM

        pre_position = target_mm - self.config.BACKLASH_DIST_MM
        if pre_position < self.config.MIN_Z_MM: pre_position = self.config.MIN_Z_MM

        self.motor.move_to(pre_position)
        self.motor.wait_for_stop()
        self.motor.move_to(target_mm)
        self.motor.wait_for_stop()
        self._current_pos = target_mm

    def close(self) -> None:
        try:
            self.motor.close()
            from ids_peak import ids_peak
            ids_peak.Library.Close()
        except Exception: pass

class MockLaserSystem(LaserSystem):
    def __init__(self, config: Config = Config()):
        print("--- Initializing MOCK Hardware ---")
        self.config = config
        self._current_exposure = self.config.START_EXPOSURE_MS
        self.z_pos = 0.0
        
        # Simple Astigmatism simulation
        self.sim_waist_z_x = 1.12
        self.sim_waist_z_y = 1.18 
        self.sim_waist_w0_x = 8.0 
        self.sim_waist_w0_y = 8.0 
        self.sim_rayleigh_x = 0.05 
        self.sim_rayleigh_y = 0.05
        self.sim_phi = np.radians(30)
        
        self.img_width = 1284
        self.img_height = 964
        x = np.arange(0, self.img_width)
        y = np.arange(0, self.img_height)
        self.xx, self.yy = np.meshgrid(x, y)
        self.cx, self.cy = self.img_width / 2, self.img_height / 2
        self.cx, self.cy = self.img_width / 2, self.img_height / 2
        
        # Pre-calculating relative coordinates for speed
        self.dx = self.xx - self.cx
        self.dy = self.yy - self.cy
        
        self._soft_limit = self.config.MAX_Z_MM

    @property
    def current_exposure(self) -> float: return self._current_exposure
    
    @property
    def current_position(self) -> float: return self.z_pos

    def set_exposure(self, exposure_ms: float) -> None:
        self._current_exposure = exposure_ms

    def get_raw_image(self) -> Optional[np.ndarray]:
        # Generate synthetic rotated elliptical Gaussian beam
        z_diff_x = self.z_pos - self.sim_waist_z_x
        z_diff_y = self.z_pos - self.sim_waist_z_y
        
        wx_um = self.sim_waist_w0_x * np.sqrt(1 + (z_diff_x / self.sim_rayleigh_x)**2)
        wy_um = self.sim_waist_w0_y * np.sqrt(1 + (z_diff_y / self.sim_rayleigh_y)**2)
        
        wx_px = wx_um / self.config.PIXEL_SIZE_UM
        wy_px = wy_um / self.config.PIXEL_SIZE_UM
        
        # Rotated coordinates
        cos_p, sin_p = np.cos(self.sim_phi), np.sin(self.sim_phi)
        U = self.dx * cos_p + self.dy * sin_p
        V = -self.dx * sin_p + self.dy * cos_p
        
        # Intensity scales with exposure and concentration (waist ratio)
        base_peak_at_waist_1ms = 12000.0
        # Energy conservation roughly: peak ~ 1/(wx*wy)
        peak_intensity = base_peak_at_waist_1ms * self._current_exposure * ((self.sim_waist_w0_x*self.sim_waist_w0_y)/(wx_um*wy_um))
        
        gaussian = peak_intensity * np.exp(-2 * (U**2 / wx_px**2 + V**2 / wy_px**2))
        
        # Reduce noise for a cleaner simulation
        noise = np.random.normal(0, 0.5, gaussian.shape) + np.random.normal(0, 0.01 * gaussian + 0.02, gaussian.shape)
        img = gaussian + noise
        
        time.sleep(0.01) # Simulate camera readout time
        return np.clip(img, 0, 255).astype(np.uint8)

    def move_motor_precise(self, target_mm: float) -> None:
        if target_mm < self.config.MIN_Z_MM: target_mm = self.config.MIN_Z_MM
        if target_mm > self._soft_limit: target_mm = self._soft_limit
        if target_mm > self.config.MAX_Z_MM: target_mm = self.config.MAX_Z_MM
        
        # Simulate travel time
        dist = abs(target_mm - self.z_pos)
        time.sleep(dist * 0.1) 
        self.z_pos = target_mm
        
    def set_soft_limit(self, limit_mm: float) -> None:
        self._soft_limit = limit_mm
        
    @property
    def soft_limit(self) -> float: return self._soft_limit

    def close(self) -> None:
        print("MOCK Hardware closed.")
