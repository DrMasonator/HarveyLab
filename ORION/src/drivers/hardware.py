"""Hardware interfaces and mock implementations."""

from __future__ import annotations

import ctypes
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ORION.config import Config

logger = logging.getLogger(__name__)


class LaserSystem(ABC):
    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_raw_image(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def move_motor_precise(self, target_mm: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_exposure(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_soft_limit(self, limit_mm: float) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def soft_limit(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_position(self) -> float:
        raise NotImplementedError


class RealLaserSystem(LaserSystem):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._current_exposure = self.config.START_EXPOSURE_MS
        self._current_pos = 0.0

        logger.info("Initializing hardware...")

        from pylablib.devices import Thorlabs
        from ids_peak import ids_peak

        try:
            self.motor = Thorlabs.KinesisMotor(self.config.MOTOR_SERIAL, scale=self.config.Z812_SCALE)
            logger.info("Motor %s connected.", self.config.MOTOR_SERIAL)
            logger.info("Homing motor...")
            self.motor.home()
            self.motor.wait_for_home()
            self._current_pos = 0.0
            self._soft_limit = self.config.MAX_Z_MM
        except Exception as exc:
            logger.exception("Failed to initialize motor")
            raise RuntimeError("Motor initialization failed") from exc

        try:
            ids_peak.Library.Initialize()
            dev_mgr = ids_peak.DeviceManager.Instance()
            dev_mgr.Update()
            if dev_mgr.Devices().empty():
                raise RuntimeError("No camera detected")

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
            logger.info("Camera stream started")
        except Exception as exc:
            logger.exception("Failed to initialize camera")
            raise RuntimeError("Camera initialization failed") from exc

    @property
    def current_exposure(self) -> float:
        return self._current_exposure

    @property
    def current_position(self) -> float:
        return float(self.motor.get_position())

    @property
    def soft_limit(self) -> float:
        return self._soft_limit

    def set_soft_limit(self, limit_mm: float) -> None:
        self._soft_limit = float(limit_mm)
        logger.info("Soft limit set to %.3f mm", limit_mm)

    def set_exposure(self, exposure_ms: float) -> None:
        try:
            exposure_us = exposure_ms * 1000.0
            self.node_map.FindNode("ExposureTime").SetValue(exposure_us)
            self._current_exposure = float(exposure_ms)
        except Exception:
            logger.exception("Failed to set exposure")

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
            logger.exception("Camera buffer read failed")
            return None
        finally:
            if buf:
                self.stream.QueueBuffer(buf)

    def move_motor_precise(self, target_mm: float) -> None:
        target_mm = float(target_mm)
        if target_mm < self.config.MIN_Z_MM:
            target_mm = self.config.MIN_Z_MM
        if target_mm > self._soft_limit:
            logger.warning(
                "Target %.3f exceeds soft limit %.3f. Clamping.", target_mm, self._soft_limit
            )
            target_mm = self._soft_limit
        if target_mm > self.config.MAX_Z_MM:
            target_mm = self.config.MAX_Z_MM

        pre_position = target_mm - self.config.BACKLASH_DIST_MM
        if pre_position < self.config.MIN_Z_MM:
            pre_position = self.config.MIN_Z_MM

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
        except Exception:
            logger.exception("Failed to close hardware cleanly")


class MockLaserSystem(LaserSystem):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._current_exposure = self.config.START_EXPOSURE_MS
        self.z_pos = 0.0

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

        self.dx = self.xx - self.cx
        self.dy = self.yy - self.cy

        self._soft_limit = self.config.MAX_Z_MM
        logger.info("Initialized mock hardware")

    @property
    def current_exposure(self) -> float:
        return self._current_exposure

    @property
    def current_position(self) -> float:
        return float(self.z_pos)

    def set_exposure(self, exposure_ms: float) -> None:
        self._current_exposure = float(exposure_ms)

    def get_raw_image(self) -> Optional[np.ndarray]:
        z_diff_x = self.z_pos - self.sim_waist_z_x
        z_diff_y = self.z_pos - self.sim_waist_z_y

        wx_um = self.sim_waist_w0_x * np.sqrt(1 + (z_diff_x / self.sim_rayleigh_x) ** 2)
        wy_um = self.sim_waist_w0_y * np.sqrt(1 + (z_diff_y / self.sim_rayleigh_y) ** 2)

        wx_px = wx_um / self.config.PIXEL_SIZE_UM
        wy_px = wy_um / self.config.PIXEL_SIZE_UM

        cos_p, sin_p = np.cos(self.sim_phi), np.sin(self.sim_phi)
        U = self.dx * cos_p + self.dy * sin_p
        V = -self.dx * sin_p + self.dy * cos_p

        base_peak_at_waist_1ms = 12000.0
        peak_intensity = base_peak_at_waist_1ms * self._current_exposure * (
            (self.sim_waist_w0_x * self.sim_waist_w0_y) / (wx_um * wy_um)
        )

        gaussian = peak_intensity * np.exp(-2 * (U**2 / wx_px**2 + V**2 / wy_px**2))
        noise = (
            np.random.normal(0, 0.5, gaussian.shape)
            + np.random.normal(0, 0.01 * gaussian + 0.02, gaussian.shape)
        )
        img = gaussian + noise

        time.sleep(0.01)
        return np.clip(img, 0, 255).astype(np.uint8)

    def move_motor_precise(self, target_mm: float) -> None:
        target_mm = float(target_mm)
        if target_mm < self.config.MIN_Z_MM:
            target_mm = self.config.MIN_Z_MM
        if target_mm > self._soft_limit:
            target_mm = self._soft_limit
        if target_mm > self.config.MAX_Z_MM:
            target_mm = self.config.MAX_Z_MM

        dist = abs(target_mm - self.z_pos)
        time.sleep(dist * 0.1)
        self.z_pos = target_mm

    def set_soft_limit(self, limit_mm: float) -> None:
        self._soft_limit = float(limit_mm)

    @property
    def soft_limit(self) -> float:
        return self._soft_limit

    def close(self) -> None:
        logger.info("Mock hardware closed")
