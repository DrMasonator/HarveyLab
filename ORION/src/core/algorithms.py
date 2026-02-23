"""Core measurement orchestration and scan routines."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ORION.config import Config
from ORION.src.core.analysis import BeamAnalyzer
from ORION.src.core.processing import ExposureController, ImageProcessor
from ORION.src.core.roi import ROIManager
from ORION.src.core.types import BeamAnalysis, EMPTY_BEAM
from ORION.src.drivers.hardware import LaserSystem

matplotlib.use("Agg")


class MeasurementOrchestrator:
    """Coordinates exposure, ROI selection, and beam analysis."""

    def __init__(
        self,
        config: Config,
        system: LaserSystem,
        analyzer: BeamAnalyzer,
        processor: ImageProcessor,
        exposure_ctrl: ExposureController,
        worker=None,
    ):
        self.config = config
        self.system = system
        self.analyzer = analyzer
        self.processor = processor
        self.exposure_ctrl = exposure_ctrl
        self.worker = worker
        self.roi = ROIManager(config)

    def _emit_live_update(self, img: np.ndarray, max_val: float) -> None:
        if not self.worker:
            return
        self.worker.new_image.emit(img)
        self.worker.stats_update.emit(
            float(max_val),
            float(self.worker.cached_d4s),
            float(self.system.current_position),
            float(self.system.current_exposure),
        )

    def _stabilize_exposure(self) -> None:
        for _ in range(10):
            if self.worker and self.worker.stop_requested:
                return
            img = self.system.get_raw_image()
            if img is None:
                continue
            proc_img, _ = self.processor.process_image(img)
            max_val = float(np.max(proc_img))
            changed = self.exposure_ctrl.handle_auto_exposure(max_val)
            self._emit_live_update(proc_img, max_val)
            if not changed:
                break
            time.sleep(0.05)

    def robust_measure_optical(
        self, *, skip_ae: bool = False, average_count: int | None = None
    ) -> BeamAnalysis:
        """
        AE stabilization + median averaging with dynamic ROI lock.
        Returns a BeamAnalysis for the median of collected frames.
        """
        if not skip_ae:
            self._stabilize_exposure()

        sample_count = average_count if average_count is not None else self.config.MEASURE_AVERAGE_COUNT
        d4s_vals: list[float] = []
        dx_vals: list[float] = []
        dy_vals: list[float] = []
        phi_vals: list[float] = []
        cx_vals: list[float] = []
        cy_vals: list[float] = []
        wpx_vals: list[float] = []
        hpx_vals: list[float] = []

        for _ in range(sample_count):
            if self.worker and self.worker.stop_requested:
                return EMPTY_BEAM
            img = self.system.get_raw_image()
            if img is None:
                continue

            proc_img, v_size = self.processor.process_image(img)

            roi_img, (x0, y0) = self._select_roi(proc_img)
            roi_max = float(np.max(roi_img)) if roi_img.size else 0.0

            res = self.analyzer.analyze_beam(roi_img, roi_max, v_size)
            d = res.d4s_eff_um

            self._emit_live_update(proc_img, roi_max)

            if d > 0:
                cx_global = float(res.centroid_x_px + x0)
                cy_global = float(res.centroid_y_px + y0)
                wpx = float(res.d4s_x_px)
                hpx = float(res.d4s_y_px)

                self.roi.on_measurement(cx_global, cy_global, wpx, hpx)
                self.roi.maybe_relock_if_near_edge(cx_global, cy_global, (x0, y0), roi_img.shape)

                d4s_vals.append(d)
                dx_vals.append(res.d4s_x_um)
                dy_vals.append(res.d4s_y_um)
                phi_vals.append(res.azimuth_deg)
                cx_vals.append(cx_global)
                cy_vals.append(cy_global)
                wpx_vals.append(wpx)
                hpx_vals.append(hpx)
            else:
                self.roi.on_miss()

        if not d4s_vals:
            return EMPTY_BEAM

        return BeamAnalysis(
            float(np.median(d4s_vals)),
            float(np.median(dx_vals)),
            float(np.median(dy_vals)),
            float(np.median(phi_vals)),
            float(np.median(cx_vals)),
            float(np.median(cy_vals)),
            float(np.median(wpx_vals)),
            float(np.median(hpx_vals)),
        )

    def _select_roi(self, proc_img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        for _ in range(3):
            roi_img, (x0, y0) = self.roi.get_crop(proc_img)
            if roi_img.size == 0:
                return roi_img, (x0, y0)

            roi_max = float(np.max(roi_img))
            clipped = False
            if roi_max > 0:
                edge_w = min(8, roi_img.shape[0] // 2, roi_img.shape[1] // 2)
                if edge_w > 0:
                    edge_top = roi_img[:edge_w, :]
                    edge_bottom = roi_img[-edge_w:, :]
                    edge_left = roi_img[:, :edge_w]
                    edge_right = roi_img[:, -edge_w:]
                    edge_peak = max(
                        float(np.max(edge_top)),
                        float(np.max(edge_bottom)),
                        float(np.max(edge_left)),
                        float(np.max(edge_right)),
                    )
                    edge_mean = max(
                        float(np.mean(edge_top)),
                        float(np.mean(edge_bottom)),
                        float(np.mean(edge_left)),
                        float(np.mean(edge_right)),
                    )
                    if (
                        edge_peak / roi_max > self.config.ROI_EDGE_PEAK_FRACTION
                        or edge_mean / roi_max > (self.config.ROI_EDGE_PEAK_FRACTION * 0.25)
                    ):
                        clipped = True

            if clipped:
                self.roi.on_border_signal()
                continue
            return roi_img, (x0, y0)

        return roi_img, (x0, y0)


class FocusOptimizer:
    def __init__(self, config: Config, system: LaserSystem, orchestrator: MeasurementOrchestrator, worker=None):
        self.config = config
        self.system = system
        self.orchestrator = orchestrator
        self.worker = worker

    def run_golden_section_search(self, start_z: float, end_z: float) -> None:
        if self.worker:
            self.worker.status_msg.emit("Starting Auto-Search...")
        if end_z <= start_z:
            if self.worker:
                self.worker.status_msg.emit("Search aborted: invalid range (end <= start).")
            return

        invphi = (np.sqrt(5) - 1) / 2
        invphi2 = (3 - np.sqrt(5)) / 2

        a, b = start_z, end_z
        h = b - a
        tol = self.config.SEARCH_TOLERANCE
        if tol <= 0 or h <= tol:
            if self.worker:
                self.worker.status_msg.emit("Search range already within tolerance.")
            return

        n_steps = int(np.ceil(np.log(tol / h) / np.log(invphi)))

        c = a + invphi2 * h
        d = a + invphi * h

        def measure(z: float) -> float:
            if self.worker:
                self.worker.status_msg.emit(f"Search: {z:.3f}mm")
            self.system.move_motor_precise(z)

            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if abs(self.system.current_position - z) < 0.005:
                    break
                time.sleep(0.05)

            res = self.orchestrator.robust_measure_optical()
            final_d4s = res.d4s_eff_um
            if self.worker and final_d4s > 0:
                self.worker.measurement_taken.emit(z, final_d4s, res.d4s_x_um, res.d4s_y_um)
            return final_d4s

        yc = measure(c)
        yd = measure(d)

        for _ in range(n_steps):
            if self.worker and self.worker.stop_requested:
                if self.worker:
                    self.worker.status_msg.emit("Search Aborted by User.")
                return

            if yc < yd:
                b = d
                d = c
                yd = yc
                h = invphi * h
                c = a + invphi2 * h
                yc = measure(c)
            else:
                a = c
                c = d
                yc = yd
                h = invphi * h
                d = a + invphi * h
                yd = measure(d)

        waist_pos = (a + b) / 2
        min_d4s = min(yc, yd)

        if self.worker:
            self.worker.status_msg.emit(
                f"Search DONE. Waist: {waist_pos:.3f}mm | {min_d4s:.1f}um"
            )
            self.system.move_motor_precise(waist_pos)
            self.worker.cached_d4s = min_d4s
            self.worker.search_finished.emit(waist_pos, min_d4s)


class BeamCharacterizer:
    def __init__(self, config: Config, system: LaserSystem, orchestrator: MeasurementOrchestrator, worker=None):
        self.config = config
        self.system = system
        self.orchestrator = orchestrator
        self.worker = worker

    def run_characterization_scan(self, z_points: list[float]) -> None:
        if not z_points:
            return

        if self.worker:
            self.worker.status_msg.emit(
                f"Starting Beam Characterization Scan ({len(z_points)} points)..."
            )

        results: list[tuple[float, float, float, float, float]] = []

        for z in z_points:
            if self.worker and self.worker.stop_requested:
                if self.worker:
                    self.worker.status_msg.emit("Scan Aborted by User.")
                break

            z = float(z)
            if self.worker:
                self.worker.status_msg.emit(f"Scanning Z: {z:.2f}mm")
            self.system.move_motor_precise(z)

            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if abs(self.system.current_position - z) < 0.005:
                    break
                time.sleep(0.05)

            vals = self.orchestrator.robust_measure_optical()
            if vals.d4s_eff_um > 0:
                results.append(
                    (z, vals.d4s_eff_um, vals.d4s_x_um, vals.d4s_y_um, vals.azimuth_deg)
                )
                if self.worker:
                    self.worker.measurement_taken.emit(z, vals.d4s_eff_um, vals.d4s_x_um, vals.d4s_y_um)

        if self.worker:
            self.worker.status_msg.emit("Scan Complete. Saving Results...")
        self._save_results(results)

    def _save_results(self, data: list[tuple[float, float, float, float, float]]) -> None:
        if not data:
            return
        project_root = Path(__file__).resolve().parents[2]
        plot_dir = project_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = plot_dir / f"beam_scan_{timestamp}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Z_Position_mm", "D4sigma_Eff_um", "D4sigma_X_um", "D4sigma_Y_um", "Azimuth_deg"])
            writer.writerows(data)

        z_arr = [row[0] for row in data]
        d4s_eff = [row[1] for row in data]
        d4s_x = [row[2] for row in data]
        d4s_y = [row[3] for row in data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(z_arr, d4s_eff, "bo-", ms=5, lw=1.5, label="Effective Diameter (D4s)")
        ax1.set_title("Beam Caustic Scan (Effective Diameter)")
        ax1.set_xlabel("Z Position (mm)")
        ax1.set_ylabel("Diameter (µm)")
        ax1.grid(True, which="both", linestyle="-", alpha=0.6)
        ax1.legend()

        ax2.plot(z_arr, d4s_x, "r.-", ms=8, lw=1.5, label="Dx (Horizontal)")
        ax2.plot(z_arr, d4s_y, "g.-", ms=8, lw=1.5, label="Dy (Vertical)")
        ax2.set_title("Beam Caustic Scan (Dx, Dy)")
        ax2.set_xlabel("Z Position (mm)")
        ax2.set_ylabel("Diameter (µm)")
        ax2.grid(True, which="both", linestyle="-", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = plot_dir / f"caustic_scan_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close(fig)

        if self.worker:
            output_msg = f"Results saved to {plot_path.name}"
            self.worker.status_msg.emit(output_msg)
            self.worker.characterization_finished.emit(str(plot_path))
