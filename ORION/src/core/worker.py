"""Background worker thread for hardware IO and measurement routines."""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

import numpy as np
from PyQt5 import QtCore

from ORION.config import Config
from ORION.src.core.algorithms import BeamCharacterizer, FocusOptimizer, MeasurementOrchestrator
from ORION.src.core.analysis import BeamAnalyzer
from ORION.src.core.processing import ExposureController, ImageProcessor
from ORION.src.core.types import BeamAnalysis, EMPTY_BEAM
from ORION.src.drivers.hardware import LaserSystem

logger = logging.getLogger(__name__)


class WorkerMode:
    LIVE = 0
    SEARCH = 1
    CHARACTERIZE = 2


class WorkerState:
    LIVE = "LIVE"
    MEASURING = "MEASURING"
    SEARCHING = "SEARCHING"
    SCANNING = "SCANNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class HardwareWorker(QtCore.QThread):
    new_image = QtCore.pyqtSignal(object)
    stats_update = QtCore.pyqtSignal(float, float, float, float)
    status_msg = QtCore.pyqtSignal(str)
    state_changed = QtCore.pyqtSignal(str)
    search_finished = QtCore.pyqtSignal(float, float)
    characterization_finished = QtCore.pyqtSignal(str)
    measurement_taken = QtCore.pyqtSignal(float, float, float, float)
    one_shot_finished = QtCore.pyqtSignal(str)
    overlay_update = QtCore.pyqtSignal(float, float, float, float, float)

    def __init__(self, system: LaserSystem, config: Config):
        super().__init__()
        self.system = system
        self.config = config
        self.running = True
        self.paused = False
        self.mode = WorkerMode.LIVE
        self.state = WorkerState.LIVE

        self.command_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cached_d4s = 0.0
        self.target_z: float | None = None
        self.stop_requested = False

        self.analyzer = BeamAnalyzer(config)
        self.processor = ImageProcessor(config)
        self.exposure_ctrl = ExposureController(config, system)
        self.orchestrator = MeasurementOrchestrator(
            config, system, self.analyzer, self.processor, self.exposure_ctrl, worker=self
        )
        self.optimizer = FocusOptimizer(config, system, self.orchestrator, worker=self)
        self.characterizer = BeamCharacterizer(config, system, self.orchestrator, worker=self)
        self.state_changed.emit(self.state)

    def _set_state(self, state: str) -> None:
        if self.state != state:
            self.state = state
            self.state_changed.emit(state)

    def _clear_queue(self) -> None:
        with self.command_queue.mutex:
            self.command_queue.queue.clear()

    def set_target_z(self, target_z: float) -> None:
        self.target_z = float(target_z)

    def start_search(self, start_z: float, end_z: float) -> None:
        self.command_queue.put(("SEARCH", (start_z, end_z)))

    def start_characterization(self, scan_points: list[float]) -> None:
        self.command_queue.put(("CHARACTERIZE", scan_points))

    def start_one_shot_measurement(self) -> None:
        self.stop_requested = False
        self.command_queue.put(("MEASURE_ONCE", None))

    def stop(self) -> None:
        self.running = False
        self.wait()

    def request_stop(self) -> None:
        self.stop_requested = True
        self._set_state(WorkerState.STOPPING)
        self._clear_queue()
        self.status_msg.emit("Stopping...")

    def run(self) -> None:
        while self.running:
            try:
                self._drain_commands()
            except Exception:
                self._set_state(WorkerState.ERROR)
                self.status_msg.emit("Worker error. Check logs for details.")
                logger.exception("Worker thread crashed")
                time.sleep(0.05)
                if self.running:
                    self._set_state(WorkerState.LIVE)

            if self.mode == WorkerMode.LIVE:
                self._live_loop()

    def _drain_commands(self) -> None:
        while not self.command_queue.empty():
            cmd, val = self.command_queue.get_nowait()
            if cmd == "SEARCH":
                self._handle_search(val)
            elif cmd == "CHARACTERIZE":
                self._handle_characterize(val)
            elif cmd == "MEASURE_ONCE":
                self._handle_measure_once()

    def _handle_search(self, val: tuple[float, float]) -> None:
        self.stop_requested = False
        self._set_state(WorkerState.SEARCHING)
        start_z, end_z = val
        self.mode = WorkerMode.SEARCH
        self.optimizer.run_golden_section_search(start_z, end_z)
        self.mode = WorkerMode.LIVE
        self._set_state(WorkerState.LIVE)
        self.target_z = self.system.current_position

    def _handle_characterize(self, scan_points: list[float]) -> None:
        self.stop_requested = False
        self._set_state(WorkerState.SCANNING)
        self.mode = WorkerMode.CHARACTERIZE
        self.characterizer.run_characterization_scan(scan_points)
        self.mode = WorkerMode.LIVE
        self._set_state(WorkerState.LIVE)
        self.target_z = self.system.current_position

    def _handle_measure_once(self) -> None:
        self.stop_requested = False
        self._set_state(WorkerState.MEASURING)
        res: BeamAnalysis = self.orchestrator.robust_measure_optical(
            skip_ae=self.config.FIND_BEAM_SKIP_AE,
            average_count=self.config.FIND_BEAM_AVERAGE_COUNT,
        )

        self.cached_d4s = res.d4s_eff_um

        msg = (
            f"D4s: {res.d4s_eff_um:.1f} um (X: {res.d4s_x_um:.1f}, Y: {res.d4s_y_um:.1f}) | "
            f"Phi: {res.azimuth_deg:.1f}°"
        )
        self.one_shot_finished.emit(msg)

        if res.d4s_eff_um > 0:
            self.overlay_update.emit(
                float(res.centroid_x_px),
                float(res.centroid_y_px),
                float(res.d4s_x_px),
                float(res.d4s_y_px),
                float(res.azimuth_deg),
            )
            self.status_msg.emit(f"Measured: {res.d4s_eff_um:.1f}um")

        self._set_state(WorkerState.LIVE)

    def _live_loop(self) -> None:
        if self.paused:
            time.sleep(0.1)
            return

        if self.target_z is not None:
            current_pos = self.system.current_position
            if abs(self.target_z - current_pos) > 0.001:
                self.system.move_motor_precise(self.target_z)
            else:
                self.target_z = None

        img = self.system.get_raw_image()
        if img is None:
            return

        proc_img, _ = self.processor.process_image(img)
        max_val = float(np.max(proc_img))
        self.exposure_ctrl.handle_auto_exposure(max_val)

        self.new_image.emit(proc_img)
        self.stats_update.emit(max_val, self.cached_d4s, self.system.current_position, self.system.current_exposure)
