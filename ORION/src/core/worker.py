import queue
import time
import numpy as np
import traceback
from PyQt5 import QtCore

from ORION.config import Config
from ORION.src.drivers.hardware import LaserSystem
from ORION.src.core.analysis import BeamAnalyzer
from ORION.src.core.processing import ImageProcessor, ExposureController
from ORION.src.core.algorithms import MeasurementOrchestrator, FocusOptimizer, BeamCharacterizer

class WorkerMode:
    LIVE = 0
    SEARCH = 1
    CHARACTERIZE = 2

class HardwareWorker(QtCore.QThread):
    # Signals
    new_image = QtCore.pyqtSignal(object) 
    stats_update = QtCore.pyqtSignal(float, float, float, float) # max_val, d4s_um, z_pos, exposure
    status_msg = QtCore.pyqtSignal(str)
    state_changed = QtCore.pyqtSignal(str)  # LIVE, MEASURING, SEARCHING, SCANNING, STOPPING, ERROR
    search_finished = QtCore.pyqtSignal(float, float) 
    characterization_finished = QtCore.pyqtSignal(str) 
    measurement_taken = QtCore.pyqtSignal(float, float, float, float) # z, d_eff, d_x, d_y
    one_shot_finished = QtCore.pyqtSignal(str) 
    overlay_update = QtCore.pyqtSignal(float, float, float, float, float) 
    
    def __init__(self, system: LaserSystem, config: Config):
        super().__init__()
        self.system = system
        self.config = config
        self.running = True
        self.paused = False
        self.mode = WorkerMode.LIVE
        self.state = "LIVE"
        
        self.command_queue = queue.Queue()
        self.cached_d4s = 0.0 
        self.target_z = None 
        self.stop_requested = False
        
        # Modular Components
        self.analyzer = BeamAnalyzer(config)
        self.processor = ImageProcessor(config)
        self.exposure_ctrl = ExposureController(config, system)
        self.orchestrator = MeasurementOrchestrator(config, system, self.analyzer, self.processor, self.exposure_ctrl, worker=self)
        self.optimizer = FocusOptimizer(config, system, self.orchestrator, worker=self)
        self.characterizer = BeamCharacterizer(config, system, self.orchestrator, worker=self)
        self.state_changed.emit(self.state)

    def _set_state(self, state: str):
        if self.state != state:
            self.state = state
            self.state_changed.emit(state)

    def set_target_z(self, target_z: float):
        self.target_z = target_z
        
    def start_search(self, start_z, end_z):
        self.command_queue.put(("SEARCH", (start_z, end_z)))

    def start_characterization(self, scan_points: list):
        self.command_queue.put(("CHARACTERIZE", scan_points))

    def start_one_shot_measurement(self):
        # Clear any prior stop request so a new measurement can proceed
        self.stop_requested = False
        self.command_queue.put(("MEASURE_ONCE", None))

    def stop(self):
        self.running = False
        self.wait()

    def request_stop(self):
        self.stop_requested = True
        self._set_state("STOPPING")
        # Clear queue
        with self.command_queue.mutex:
            self.command_queue.queue.clear()
        self.status_msg.emit("Stopping...")

    def run(self):
        while self.running:
            try:
                while not self.command_queue.empty():
                    cmd, val = self.command_queue.get_nowait()
                    if cmd == "SEARCH":
                        self.stop_requested = False
                        self._set_state("SEARCHING")
                        start_z, end_z = val
                        self.mode = WorkerMode.SEARCH
                        self.optimizer.run_golden_section_search(start_z, end_z)
                        self.mode = WorkerMode.LIVE
                        self._set_state("LIVE")
                        self.target_z = self.system.current_position
                    elif cmd == "CHARACTERIZE":
                        self.stop_requested = False
                        self._set_state("SCANNING")
                        scan_points = val
                        self.mode = WorkerMode.CHARACTERIZE
                        self.characterizer.run_characterization_scan(scan_points)
                        self.mode = WorkerMode.LIVE
                        self._set_state("LIVE")
                        self.target_z = self.system.current_position
                    elif cmd == "MEASURE_ONCE":
                        self.stop_requested = False
                        self._set_state("MEASURING")
                        res = self.orchestrator.robust_measure_optical(
                            skip_ae=self.config.FIND_BEAM_SKIP_AE,
                            average_count=self.config.FIND_BEAM_AVERAGE_COUNT,
                        )
                        d, dx, dy, phi = res[0], res[1], res[2], res[3]
                        cx, cy, wpx, hpx = res[4], res[5], res[6], res[7]
                        
                        self.cached_d4s = d
                        
                        msg = f"D4s: {d:.1f} um (X: {dx:.1f}, Y: {dy:.1f}) | Phi: {phi:.1f}°"
                        self.one_shot_finished.emit(msg) 
                        if d > 0:
                            self.overlay_update.emit(float(cx), float(cy), float(wpx), float(hpx), float(phi))
                            self.status_msg.emit(f"Measured: {d:.1f}um (Overlay Emitted)")
                        self._set_state("LIVE")
            except Exception as exc:
                # Keep the worker alive, but never swallow failures silently.
                self._set_state("ERROR")
                self.status_msg.emit(f"Worker error: {exc}")
                traceback.print_exc()
                time.sleep(0.05)
                if self.running:
                    self._set_state("LIVE")

            if self.mode == WorkerMode.LIVE:
                if self.paused:
                    time.sleep(0.1)
                    continue

                if self.target_z is not None:
                     current_pos = self.system.current_position
                     if abs(self.target_z - current_pos) > 0.001:
                         self.system.move_motor_precise(self.target_z)
                     else:
                         self.target_z = None 
                
                current_hw_pos = self.system.current_position
                img = self.system.get_raw_image()
                if img is None: continue

                proc_img, v_size = self.processor.process_image(img)
                max_val = np.max(proc_img)
                self.exposure_ctrl.handle_auto_exposure(max_val)

                # Emit updates
                self.new_image.emit(proc_img)
                self.stats_update.emit(float(max_val), self.cached_d4s, current_hw_pos, self.system.current_exposure)
