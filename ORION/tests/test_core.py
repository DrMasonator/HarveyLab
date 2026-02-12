import unittest
from pathlib import Path
import tempfile
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ORION.config import Config
from ORION.src.core.processing import ImageProcessor, ExposureController
from ORION.src.core.analysis import BeamAnalyzer
from ORION.src.core.roi import ROIManager
from ORION.src.core.algorithms import MeasurementOrchestrator, FocusOptimizer


class FakeLaserSystem:
    def __init__(self, image: np.ndarray, exposure_ms: float = 1.0):
        self._image = image
        self._exposure = exposure_ms
        self._position = 0.0
        self._soft_limit = 100.0

    def set_exposure(self, exposure_ms: float) -> None:
        self._exposure = exposure_ms

    def get_raw_image(self):
        return self._image

    def move_motor_precise(self, target_mm: float) -> None:
        self._position = float(target_mm)

    def close(self) -> None:
        return None

    @property
    def current_exposure(self) -> float:
        return self._exposure

    def set_soft_limit(self, limit_mm: float) -> None:
        self._soft_limit = float(limit_mm)

    @property
    def soft_limit(self) -> float:
        return self._soft_limit

    @property
    def current_position(self) -> float:
        return self._position


def make_gaussian(h=64, w=64, cx=32.0, cy=32.0, sigma=3.0, peak=200.0):
    y, x = np.mgrid[0:h, 0:w]
    g = peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return np.clip(g, 0, 255).astype(np.uint8)


class TestConfig(unittest.TestCase):
    def test_save_load_roundtrip(self):
        cfg = Config()
        cfg.START_EXPOSURE_MS = 2.5
        cfg.TARGET_BRIGHTNESS_MAX = 210
        cfg.BAYER_MODE = "GREEN"

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            cfg.save(path)
            loaded = Config.load(path)

        self.assertAlmostEqual(loaded.START_EXPOSURE_MS, 2.5, places=3)
        self.assertEqual(loaded.TARGET_BRIGHTNESS_MAX, 210)
        self.assertEqual(loaded.BAYER_MODE, "GREEN")


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.img = np.arange(16, dtype=np.uint8).reshape(4, 4)

    def test_raw_mode(self):
        self.cfg.BAYER_MODE = "RAW"
        proc, px = ImageProcessor(self.cfg).process_image(self.img)
        self.assertEqual(px, self.cfg.PIXEL_SIZE_UM)
        self.assertEqual(proc.shape, self.img.shape)
        np.testing.assert_array_equal(proc, self.img)

    def test_red_mode(self):
        self.cfg.BAYER_MODE = "RED"
        proc, px = ImageProcessor(self.cfg).process_image(self.img)
        expected = self.img[1::2, 0::2]
        np.testing.assert_array_equal(proc, expected)
        self.assertAlmostEqual(px, self.cfg.PIXEL_SIZE_UM * 2)

    def test_green_mode(self):
        self.cfg.BAYER_MODE = "GREEN"
        proc, px = ImageProcessor(self.cfg).process_image(self.img)
        g1 = self.img[0::2, 0::2].astype(np.uint16)
        g2 = self.img[1::2, 1::2].astype(np.uint16)
        expected = ((g1 + g2) // 2).astype(np.uint8)
        np.testing.assert_array_equal(proc, expected)
        self.assertAlmostEqual(px, self.cfg.PIXEL_SIZE_UM)

    def test_blue_mode(self):
        self.cfg.BAYER_MODE = "BLUE"
        proc, px = ImageProcessor(self.cfg).process_image(self.img)
        expected = self.img[0::2, 1::2]
        np.testing.assert_array_equal(proc, expected)
        self.assertAlmostEqual(px, self.cfg.PIXEL_SIZE_UM * 2)


class TestExposureController(unittest.TestCase):
    def test_saturation_reduces_exposure(self):
        cfg = Config()
        img = make_gaussian()
        system = FakeLaserSystem(img, exposure_ms=10.0)
        ctrl = ExposureController(cfg, system)
        changed = ctrl.handle_auto_exposure(cfg.ABSOLUTE_SATURATION)
        self.assertTrue(changed)
        self.assertAlmostEqual(system.current_exposure, 5.0, places=3)

    def test_low_signal_increases_exposure(self):
        cfg = Config()
        img = make_gaussian()
        system = FakeLaserSystem(img, exposure_ms=10.0)
        ctrl = ExposureController(cfg, system)
        changed = ctrl.handle_auto_exposure(cfg.LOW_SIGNAL_THRESHOLD - 1)
        self.assertTrue(changed)
        self.assertAlmostEqual(system.current_exposure, 15.0, places=3)

    def test_high_signal_reduces_exposure(self):
        cfg = Config()
        img = make_gaussian()
        system = FakeLaserSystem(img, exposure_ms=10.0)
        ctrl = ExposureController(cfg, system)
        max_val = cfg.TARGET_BRIGHTNESS_MAX + 20
        changed = ctrl.handle_auto_exposure(max_val)
        self.assertTrue(changed)
        self.assertLess(system.current_exposure, 10.0)


class TestROIManager(unittest.TestCase):
    def test_miss_relocks(self):
        cfg = Config()
        roi = ROIManager(cfg)
        roi.locked = True
        roi.force_relock = False
        roi.on_miss()
        roi.on_miss()
        roi.on_miss()
        self.assertTrue(roi.force_relock)
        self.assertFalse(roi.locked)

    def test_border_signal_expands(self):
        cfg = Config()
        roi = ROIManager(cfg)
        prev_w, prev_h = roi.roi_w, roi.roi_h
        roi.on_border_signal()
        self.assertGreaterEqual(roi.roi_w, prev_w)
        self.assertGreaterEqual(roi.roi_h, prev_h)

    def test_full_frame_crop(self):
        cfg = Config()
        roi = ROIManager(cfg)
        roi.use_full_frame = True
        img = np.zeros((20, 30), dtype=np.uint8)
        crop, (x0, y0) = roi.get_crop(img)
        self.assertEqual((x0, y0), (0, 0))
        self.assertEqual(crop.shape, img.shape)


class TestBeamAnalyzer(unittest.TestCase):
    def test_gaussian_beam_size(self):
        cfg = Config()
        cfg.NOISE_CUTOFF_PERCENT = 0.1
        analyzer = BeamAnalyzer(cfg)
        sigma = 3.0
        img = make_gaussian(h=64, w=64, cx=32, cy=32, sigma=sigma, peak=200)
        d4s, dx, dy, phi, cx, cy, *_ = analyzer.analyze_beam(img, float(img.max()), cfg.PIXEL_SIZE_UM)
        self.assertGreater(d4s, 0.0)
        expected = 4.0 * sigma * cfg.PIXEL_SIZE_UM
        self.assertTrue(0.5 * expected <= d4s <= 1.8 * expected)
        self.assertAlmostEqual(cx, 32.0, delta=2.0)
        self.assertAlmostEqual(cy, 32.0, delta=2.0)

    def test_empty_image_returns_zero(self):
        cfg = Config()
        analyzer = BeamAnalyzer(cfg)
        img = np.zeros((32, 32), dtype=np.uint8)
        res = analyzer.analyze_beam(img, 0.0, cfg.PIXEL_SIZE_UM)
        self.assertEqual(res[0], 0.0)


class TestMeasurementOrchestrator(unittest.TestCase):
    def test_measurement_returns_nonzero(self):
        cfg = Config()
        cfg.BAYER_MODE = "RAW"
        cfg.ROI_MIN_SIZE_PX = 16
        cfg.ROI_MAX_SIZE_PX = 64
        cfg.ROI_SEARCH_BLOCK = 8
        img = make_gaussian(h=64, w=64, cx=32, cy=32, sigma=3.0, peak=220)
        system = FakeLaserSystem(img, exposure_ms=1.0)
        analyzer = BeamAnalyzer(cfg)
        processor = ImageProcessor(cfg)
        exposure = ExposureController(cfg, system)
        orch = MeasurementOrchestrator(cfg, system, analyzer, processor, exposure, worker=None)

        res = orch.robust_measure_optical(skip_ae=True, average_count=3)
        self.assertGreater(res[0], 0.0)
        self.assertAlmostEqual(res[4], 32.0, delta=2.0)
        self.assertAlmostEqual(res[5], 32.0, delta=2.0)


class TestFocusOptimizer(unittest.TestCase):
    def test_focus_optimizer_moves_to_min(self):
        cfg = Config()
        cfg.SEARCH_TOLERANCE = 0.1
        img = make_gaussian(h=32, w=32, cx=16, cy=16, sigma=2.0, peak=200)
        system = FakeLaserSystem(img, exposure_ms=1.0)

        class StubOrchestrator:
            def __init__(self, sys):
                self.sys = sys

            def robust_measure_optical(self, *args, **kwargs):
                # Convex function with minimum at z=2.0
                z = self.sys.current_position
                val = (z - 2.0) ** 2 + 1.0
                return (val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        optimizer = FocusOptimizer(cfg, system, StubOrchestrator(system), worker=None)
        optimizer.run_golden_section_search(0.0, 6.0)
        self.assertAlmostEqual(system.current_position, 2.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
