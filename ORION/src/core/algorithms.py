import numpy as np
import time
import os
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ORION.config import Config
from ORION.src.drivers.hardware import LaserSystem
from ORION.src.core.analysis import BeamAnalyzer
from ORION.src.core.processing import ImageProcessor, ExposureController

class MeasurementOrchestrator:
    def __init__(self, config: Config, system: LaserSystem, 
                 analyzer: BeamAnalyzer, processor: ImageProcessor, 
                 exposure_ctrl: ExposureController, worker=None):
        self.config = config
        self.system = system
        self.analyzer = analyzer
        self.processor = processor
        self.exposure_ctrl = exposure_ctrl
        self.worker = worker # Reference for signal emitting if needed

    def robust_measure_optical(self) -> tuple:
        """
        AE stabilization + Median Averaging.
        Returns (d4s_eff, d4s_x, d4s_y, azimuth, cx, cy, w_px, h_px).
        """
        # Stabilize Exposure
        for _ in range(10): 
             if self.worker and self.worker.stop_requested: return (0,0,0,0,0,0,0,0)
             img = self.system.get_raw_image()
             if img is None: continue
             proc_img, v_size = self.processor.process_image(img)
             mx = np.max(proc_img)
             changed = self.exposure_ctrl.handle_auto_exposure(mx)
             
             if self.worker:
                 self.worker.new_image.emit(proc_img)
                 self.worker.stats_update.emit(float(mx), float(self.worker.cached_d4s), 
                                              float(self.system.current_position), 
                                              float(self.system.current_exposure))
                                   
             if not changed: break
             time.sleep(0.05)

        # Acquire Data
        l_d4s, l_dx, l_dy, l_phi = [], [], [], []
        l_cx, l_cy, l_wpx, l_hpx = [], [], [], []

        for _ in range(self.config.MEASURE_AVERAGE_COUNT):
            if self.worker and self.worker.stop_requested: return (0,0,0,0,0,0,0,0)
            img = self.system.get_raw_image()
            if img is None: continue
            proc_img, v_size = self.processor.process_image(img)
            mx = np.max(proc_img)
            res = self.analyzer.analyze_beam(proc_img, mx, v_size)
            d = res[0]
            
            if self.worker:
                self.worker.new_image.emit(proc_img)
                self.worker.stats_update.emit(float(mx), float(d), 
                                             float(self.system.current_position), 
                                             float(self.system.current_exposure))
            
            if d > 0:
                l_d4s.append(d); l_dx.append(res[1]); l_dy.append(res[2]); l_phi.append(res[3])
                l_cx.append(res[4]); l_cy.append(res[5]); l_wpx.append(res[6]); l_hpx.append(res[7])
        
        if not l_d4s:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (np.median(l_d4s), np.median(l_dx), np.median(l_dy), np.median(l_phi),
                np.median(l_cx), np.median(l_cy), np.median(l_wpx), np.median(l_hpx))

class FocusOptimizer:
    def __init__(self, config: Config, system: LaserSystem, orchestrator: MeasurementOrchestrator, worker=None):
        self.config = config
        self.system = system
        self.orchestrator = orchestrator
        self.worker = worker

    def run_golden_section_search(self, start_z, end_z):
        if self.worker: self.worker.status_msg.emit("Starting Auto-Search...")
        invphi = (np.sqrt(5) - 1) / 2
        invphi2 = (3 - np.sqrt(5)) / 2
        
        a, b = start_z, end_z
        h = b - a
        tol = self.config.SEARCH_TOLERANCE
        n_steps = int(np.ceil(np.log(tol / h) / np.log(invphi)))

        c = a + invphi2 * h
        d = a + invphi * h

        def measure(z):
            if self.worker: self.worker.status_msg.emit(f"Search: {z:.3f}mm")
            self.system.move_motor_precise(z)
            
            start_wait = time.time()
            arrived = False
            while time.time() - start_wait < 2.0:
                if abs(self.system.current_position - z) < 0.005: 
                    arrived = True; break
                time.sleep(0.05)
            
            res = self.orchestrator.robust_measure_optical()
            final_d4s = res[0]
            
            if self.worker and final_d4s > 0:
                self.worker.measurement_taken.emit(z, final_d4s, res[1], res[2])
            return final_d4s

        yc = measure(c)
        yd = measure(d)

        for _ in range(n_steps):
            if self.worker and self.worker.stop_requested:
                if self.worker: self.worker.status_msg.emit("Search Aborted by User.")
                return

            if yc < yd:
                b = d; d = c; yd = yc; h = invphi * h; c = a + invphi2 * h
                yc = measure(c)
            else:
                a = c; c = d; yc = yd; h = invphi * h; d = a + invphi * h
                yd = measure(d)
                
        waist_pos = (a + b) / 2
        min_d4s = min(yc, yd)
        
        if self.worker:
            self.worker.status_msg.emit(f"Search DONE. Waist: {waist_pos:.3f}mm | {min_d4s:.1f}um")
            self.system.move_motor_precise(waist_pos)
            self.worker.cached_d4s = min_d4s
            self.worker.search_finished.emit(waist_pos, min_d4s)

class BeamCharacterizer:
    def __init__(self, config: Config, system: LaserSystem, orchestrator: MeasurementOrchestrator, worker=None):
        self.config = config
        self.system = system
        self.orchestrator = orchestrator
        self.worker = worker

    def run_characterization_scan(self, z_points: list):
        if not z_points: return
        
        start_z = z_points[0]
        end_z = z_points[-1]
        if self.worker: self.worker.status_msg.emit(f"Starting Beam Characterization Scan ({len(z_points)} points)...")
        
        results = []
        
        for z in z_points:
            if self.worker and self.worker.stop_requested:
                if self.worker: self.worker.status_msg.emit("Scan Aborted by User.")
                break
                
            z = float(z)
            if self.worker: self.worker.status_msg.emit(f"Scanning Z: {z:.2f}mm")
            self.system.move_motor_precise(z)
            
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if abs(self.system.current_position - z) < 0.005: break
                time.sleep(0.05)
            
            vals = self.orchestrator.robust_measure_optical()
            val_d = vals[0]
            val_dx = vals[1]
            val_dy = vals[2]
            
            if val_d > 0:
                results.append((z, vals[0], vals[1], vals[2], vals[3]))
                if self.worker: self.worker.measurement_taken.emit(z, val_d, val_dx, val_dy)

        if self.worker: self.worker.status_msg.emit("Scan Complete. Saving Results...")
        self.save_results(results)

    def save_results(self, data):
        if not data: return
        plot_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = os.path.join(plot_dir, f"beam_scan_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Z_Position_mm", "D4sigma_Eff_um", "D4sigma_X_um", "D4sigma_Y_um", "Azimuth_deg"])
            writer.writerows(data)

        z_arr = [row[0] for row in data]
        d4s_eff = [row[1] for row in data]
        d4s_x = [row[2] for row in data]
        d4s_y = [row[3] for row in data]
        
        # Create Subplots: 2 rows, 1 col
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 1. Effective Radius (or Diameter?)
        # User image says "Radius (um)" but our data is Diameter (D4sigma).
        # We will plot D4sigma for now but label it clearly.
        # Blue line with circle markers
        ax1.plot(z_arr, d4s_eff, 'bo-', ms=5, lw=1.5, label='Effective Diameter (D4s)')
        ax1.set_title("Beam Caustic Scan (Effective Diameter)")
        ax1.set_xlabel("Z Position (mm)")
        ax1.set_ylabel("Diameter (µm)")
        ax1.grid(True, which='both', linestyle='-', alpha=0.6)
        ax1.legend()
        
        # 2. X/Y Diameters
        # Red and Green lines with dot markers
        ax2.plot(z_arr, d4s_x, 'r.-', ms=8, lw=1.5, label='Dx (Horizontal)')
        ax2.plot(z_arr, d4s_y, 'g.-', ms=8, lw=1.5, label='Dy (Vertical)')
        ax2.set_title("Beam Caustic Scan (Dx, Dy)")
        ax2.set_xlabel("Z Position (mm)")
        ax2.set_ylabel("Diameter (µm)")
        ax2.grid(True, which='both', linestyle='-', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"caustic_scan_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        
        if self.worker:
            output_msg = f"Results saved to {os.path.basename(plot_path)}"
            self.worker.status_msg.emit(output_msg)
            self.worker.characterization_finished.emit(plot_path)
