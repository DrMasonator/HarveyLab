import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ... (Imports) ...
from ORION.config import Config
from ORION.src.drivers.hardware import LaserSystem
from ORION.src.core.worker import HardwareWorker
from ORION.src.ui.theme import HEX_BG_DARK

from ORION.src.ui.widgets.readouts import ReadoutWidget
from ORION.src.ui.widgets.motor_control import MotorControlWidget

class ImagingPage(QtWidgets.QWidget):
    settings_requested = QtCore.pyqtSignal()
    def __init__(self, systemList: list, config: Config):
        # systemList: [LaserSystem, HardwareWorker] passed from MainWindow
        super().__init__()
        self.system = systemList[0]
        self.worker = systemList[1]
        self.config = config
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # TOP AREA: Splitter (Image | Plots)
        self.top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # 1. Image View Area (Left)
        self.imv_container = QtWidgets.QWidget()
        # self.imv_container.setStyleSheet("background-color: #1a1a1a;") # HANDLED BY THEME
        imv_layout = QtWidgets.QVBoxLayout(self.imv_container)
        imv_layout.setContentsMargins(0, 0, 0, 0)
        
        self.imv = pg.ImageView()
        self.imv.ui.histogram.hide() 
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        
        # PRO-ACTIVE DISABLE of all auto-range garbage
        self.imv.view.setAspectLocked(True)
        self.imv.view.disableAutoRange()
        self.imv.view.setMenuEnabled(False)
        
        self.vb = self.imv.getView()
        self.vb.setBackgroundColor(HEX_BG_DARK)
        
        # Sensor Frame
        self.sensor_frame = QtWidgets.QGraphicsRectItem()
        pen = QtGui.QPen(QtGui.QColor(150, 150, 150), 1)
        pen.setCosmetic(True)
        self.sensor_frame.setPen(pen)
        self.sensor_frame.setZValue(10)
        self.vb.addItem(self.sensor_frame)
        
        imv_layout.addWidget(self.imv)
        
        # 2. Side Plots Area (Right)
        self.plot_container = pg.GraphicsLayoutWidget()
        self.plot_container.setBackground(HEX_BG_DARK)
        self.plot_container.setMinimumWidth(240)
        self.plot_container.setMaximumWidth(420)
        
        # Major Axis Plot
        self.p_major = self.plot_container.addPlot(title="Major Axis")
        self.p_major.showGrid(x=True, y=True, alpha=0.3)
        self.p_major.setLabel('left', 'Intensity')
        self.p_major.setLabel('bottom', 'Distance from Center', units='um')
        self.curve_major_raw = self.p_major.plot(pen=pg.mkPen('w', width=1), symbol='o', symbolSize=3, symbolBrush='w')
        self.curve_major_fit = self.p_major.plot(pen=pg.mkPen('#00aaff', width=2))
        
        self.plot_container.nextRow()
        
        # Minor Axis Plot
        self.p_minor = self.plot_container.addPlot(title="Minor Axis")
        self.p_minor.showGrid(x=True, y=True, alpha=0.3)
        self.p_minor.setLabel('left', 'Intensity')
        self.p_minor.setLabel('bottom', 'Distance from Center', units='um')
        self.curve_minor_raw = self.p_minor.plot(pen=pg.mkPen('w', width=1), symbol='o', symbolSize=3, symbolBrush='w')
        self.curve_minor_fit = self.p_minor.plot(pen=pg.mkPen('#ffaa00', width=2))
        
        self.top_splitter.addWidget(self.imv_container)
        self.top_splitter.addWidget(self.plot_container)
        self.top_splitter.setStretchFactor(0, 3) # Image gets 3 parts
        self.top_splitter.setStretchFactor(1, 1) # Plots get 1 part
        self.top_splitter.setSizes([900, 320])
        
        layout.addWidget(self.top_splitter, stretch=16)

        # 3. Bottom Panel (Readouts | Controls | Image Menu)
        bottom_panel = QtWidgets.QHBoxLayout()
        layout.addLayout(bottom_panel, stretch=5) # 5 units high (~1/4 of total height)
        
        self.readouts = ReadoutWidget()
        self.controls = MotorControlWidget()
        
        self.readouts.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        bottom_panel.addWidget(self.readouts, 1)
        bottom_panel.addWidget(self.controls, 2)
        
        # 3. Connections
        self.worker.new_image.connect(self.update_image)
        self.worker.stats_update.connect(self.update_stats)
        self.worker.status_msg.connect(self.update_status_msg)
        self.worker.state_changed.connect(self.on_state_changed)
        self.worker.measurement_taken.connect(self.add_plot_point)
        self.worker.one_shot_finished.connect(self.on_one_shot_finished)
        self.worker.overlay_update.connect(self.draw_beam_overlay)
        
        # Widget Specific Connections
        self.controls.move_requested.connect(self.on_move_to_requested)
        self.controls.find_beam_requested.connect(self.on_find_beam)
        self.controls.find_focus_requested.connect(self.on_find_focus_clicked)
        self.controls.measure_requested.connect(self.on_measure_clicked)
        self.controls.step_requested.connect(self.on_step_requested)
        self.controls.set_safe_requested.connect(self.on_set_safe)
        self.controls.reset_safe_requested.connect(self.on_reset_safe)
        self.controls.settings_requested.connect(lambda: self.settings_requested.emit())
        
        # Connect indicator toggle
        self.controls.check_overlay.stateChanged.connect(self.on_overlay_toggled)

        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self, activated=self.on_find_beam)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+M"), self, activated=self.on_measure_clicked)

        self.search_points_z = []
        self.search_points_d = []
        self.last_d4s_um = 0.0 
        self.current_z_read = 0.0
        self.target_z = 0.0
        self.safe_limit = 12.0
        self._is_safe_set = False
        
        self.beam_overlay = None
        self.last_beam_params = None
        self._current_state = "LIVE"
        self.set_ui_state("LIVE")

    def on_overlay_toggled(self, checked):
        if not self.beam_overlay:
            # If we have cached params but no overlay (e.g. it was cleared), redraw it but don't auto-center
            if checked and self.last_beam_params:
                cx, cy, w, h, angle = self.last_beam_params
                self.draw_beam_overlay(cx, cy, w, h, angle, auto_zoom=False)
            return

        if checked:
            self.beam_overlay.show()
        else:
            self.beam_overlay.hide()

    def update_image(self, img):
        if img is None:
            return
            
        h, w = img.shape
        # Initialize bounds ONLY once or if shape changes
        if not hasattr(self, '_prev_shape') or self._prev_shape != (h, w):
            self.sensor_frame.setRect(0, 0, w, h)
            
            # Setup limits ONCE. We do NOT touch this again unless size changes.
            margin = 5.0 # Huge margin to ensure we never hit a wall
            self.vb.setLimits(xMin=-w*margin, xMax=w*(1+margin), 
                             yMin=-h*margin, yMax=h*(1+margin),
                             minXRange=1.0, minYRange=1.0,
                             maxXRange=w*10, maxYRange=h*10) # Basically unlimited
            
            self.vb.setRange(QtCore.QRectF(0, 0, w, h), padding=0)
            self._prev_shape = (h, w)
            self.vb.disableAutoRange() # Ensure this is off forever

        # Just update the data. Do NOT touch the view range.
        self.imv.setImage(img, autoRange=False, autoLevels=False, levels=(0, 255))
        
    def update_stats(self, max_val, d4s, z_pos, exposure):
        self.current_z_read = z_pos
        self.readouts.update_stats(max_val, d4s, z_pos, exposure)
        if d4s > 0:
            self.last_d4s_um = d4s

    def update_status_msg(self, msg):
        self.readouts.update_status(msg)

    def on_state_changed(self, state: str):
        self.set_ui_state(state)

    def set_ui_state(self, state: str):
        self._current_state = state
        busy = state in {"MEASURING", "SEARCHING", "SCANNING", "STOPPING"}
        reason = f"Disabled while {state.lower()}." if busy else ""
        self.controls.btn_find_beam.setEnabled(not busy)
        self.controls.btn_find_beam.setToolTip(reason or "Measure once and center the beam in view")
        if state == "SEARCHING":
            self.controls.btn_find_focus.setEnabled(True)
            self.controls.btn_find_focus.setText("Stop Focus")
            self.controls.btn_find_focus.setToolTip("Stop the current autofocus search")
        elif state == "STOPPING":
            self.controls.btn_find_focus.setEnabled(False)
            self.controls.btn_find_focus.setText("Stopping...")
            self.controls.btn_find_focus.setToolTip("Stopping autofocus search")
        else:
            self.controls.btn_find_focus.setEnabled(not busy)
            self.controls.btn_find_focus.setText("Find Focus")
            self.controls.btn_find_focus.setToolTip(reason or "Run autofocus search between 0 and Safe Z")
        self.controls.btn_measure.setEnabled(not busy)
        self.controls.btn_measure.setToolTip(reason or "Take one beam measurement")
        self.controls.btn_back.setEnabled(not busy)
        self.controls.btn_back.setToolTip(reason or "Jog backward by selected Z-step")
        self.controls.btn_fwd.setEnabled(not busy)
        self.controls.btn_fwd.setToolTip(reason or "Jog forward by selected Z-step")
        self.controls.btn_set_safe.setEnabled(not busy)
        self.controls.btn_set_safe.setToolTip(reason or "Set current position as max safe travel")
        self.controls.btn_reset_safe.setEnabled(not busy)
        self.controls.btn_reset_safe.setToolTip(reason or "Reset Safe Limit")
        self.controls.txt_move_z.setEnabled(not busy)
        self.controls.txt_move_z.setToolTip(reason or "Target Z position in mm")
        self.controls.combo_step.setEnabled(not busy)

    def apply_config_update(self):
        # Reset ROI sizing to reflect any settings changes
        try:
            self.worker.orchestrator.roi.reset()
        except Exception:
            pass
        self.update_status_msg("Settings applied.")

    def on_move_to_requested(self, val):
        if val < 0: val = 0.0
        # Simple safety
        if val > self.safe_limit:
            val = self.safe_limit
        self.worker.set_target_z(val)

    def on_step_requested(self, direction):
        # Determine step size from combo
        text = self.controls.combo_step.currentText()
        if "um" in text:
            step = float(text.replace("um", "")) / 1000.0
        else:
            step = float(text.replace("mm", ""))
        
        new_z = self.current_z_read + (direction * step)
        self.on_move_to_requested(new_z)

    def on_find_beam(self):
        """Triggers a measurement and requests the UI to center/zoom on the resulting centroid."""
        self.update_status_msg("Searching for beam centroid...")
        self._auto_center_requested = True
        self.worker.start_one_shot_measurement()
        
    def on_find_focus_clicked(self):
        if self._current_state == "SEARCHING":
            self.worker.request_stop()
            return

        if not self._is_safe_set:
            QtWidgets.QMessageBox.warning(self, "Find Focus", "You MUST set a Safe Z limit first.\nMove to a safe maximum Z-dist and click 'Set safe Z'.")
            return
        
        self.update_status_msg("Starting Auto-Focus Search (Golden Ratio)...")
        # Assume 0 is the home/min position
        self.worker.start_search(0.0, self.safe_limit)

    def on_measure_clicked(self):
        self._auto_center_requested = False # Normal measurement doesn't jump view
        self.worker.start_one_shot_measurement()

    def on_set_safe(self):
        # We now set the GLOBAL soft limit on the driver
        self.safe_limit = self.current_z_read
        self.worker.system.set_soft_limit(self.safe_limit)
        self._is_safe_set = True
        self.controls.set_safe_z_display(self.safe_limit)

    def on_reset_safe(self):
        self.safe_limit = 12.0 # Default max travel
        self.worker.system.set_soft_limit(self.config.MAX_Z_MM)
        self._is_safe_set = False
        self.controls.lbl_safe_val.setText("---")

    def on_one_shot_finished(self, msg):
        self.readouts.update_measurement(msg)
        # Update profiles if we have valid params
        if "D4s: 0.0" not in msg and self.last_beam_params:
            cx, cy, w, h, angle = self.last_beam_params
            self.update_profiles(cx, cy, w, h, angle)

        # If Find Beam is active, only clear the flag on failure.
        # On success, overlay_update will arrive right after this and perform centering.
        if getattr(self, '_auto_center_requested', False):
            if "D4s: 0.0" in msg:
                QtWidgets.QMessageBox.warning(self, "Find Beam", "Could not find a beam in the current frame. Check alignment or exposure.")
                self._auto_center_requested = False
        else:
            self.update_status_msg(msg)

    def add_plot_point(self, z, d, dx, dy):
        pass # Handle in Caustic page if needed

    def draw_beam_overlay(self, cx, cy, w, h, angle_deg, auto_zoom=True):
        # Swap axes if angle > 45 to match visual rotation
        draw_w, draw_h = w, h
        if abs(angle_deg) > 45.0:
            draw_w, draw_h = h, w

        self.last_beam_params = (cx, cy, w, h, angle_deg)

        # 1. AUTO-CENTER / AUTO-ZOOM (must run regardless of overlay visibility)
        if auto_zoom and getattr(self, '_auto_center_requested', False):
            self._auto_center_requested = False
            self.center_beam_in_view(cx, cy, draw_w, draw_h)

        # 2. Clear old overlay
        if self.beam_overlay:
            try:
                self.vb.removeItem(self.beam_overlay)
            except: pass
            self.beam_overlay = None

        if not self.controls.check_overlay.isChecked():
            return

        if draw_w <= 0 or draw_h <= 0:
            return

        # 3. Draw Green Ellipse
        rect = QtCore.QRectF(-draw_w/2, -draw_h/2, draw_w, draw_h)
        self.beam_overlay = QtWidgets.QGraphicsEllipseItem(rect)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 120)) # Bright Green
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.beam_overlay.setPen(pen)
        
        tr = QtGui.QTransform()
        tr.translate(cx + 0.5, cy + 0.5) # Shift to pixel center
        tr.rotate(angle_deg) 
        self.beam_overlay.setTransform(tr)
        self.vb.addItem(self.beam_overlay)

        # 4. Update Profile Plots
        self.update_profiles(cx, cy, draw_w, draw_h, angle_deg)

    def center_beam_in_view(self, cx, cy, w, h):
        # Fit roughly +/-3 sigma around beam center (about 6 sigma total view).
        # Note: D4sigma diameter = 4*sigma, so 6*sigma span ~= 1.5 * D4sigma.
        beam_span = max(float(w), float(h), 1.0)
        span = max(120.0, beam_span * 1.5)
        x_min, x_max = cx - span / 2.0, cx + span / 2.0
        y_min, y_max = cy - span / 2.0, cy + span / 2.0

        def exec_zoom():
            self.vb.setAspectLocked(False)
            self.vb.enableAutoRange(enable=False)
            self.vb.setXRange(x_min, x_max, padding=0.02)
            self.vb.setYRange(y_min, y_max, padding=0.02)
            self.vb.setAspectLocked(True)
            self.update_status_msg(f"Beam centered: {cx:.1f}, {cy:.1f}")

        QtCore.QTimer.singleShot(0, exec_zoom)
    
    def update_profiles(self, cx, cy, w, h, angle_deg):
        """
        Extracts INTEGRATED cross-sections.
        Simulates a slit scan by extracting a 2D rotated region around the beam
        and summing along the perpendicular axis.
        """
        img = self.imv.image
        if img is None: return

        # 1. Define integration region
        # We want to go out ~2x radius (4x sigma) on each side -> 4x width total view
        # But we only need to integrate over the "beam width" for the signal.
        # Actually, standard is to integrate over the whole sensor, or at least 3*w.
        # Let's extract a square region of size max(w, h)*3 to be safe and simple
        
        region_w = w * 4.0 if w > 0 else 50
        region_h = h * 4.0 if h > 0 else 50

        # Clamp integration region to the image size to avoid huge allocations
        img_h, img_w = img.shape
        region_w = min(region_w, float(img_w))
        region_h = min(region_h, float(img_h))
        if region_w < 5 or region_h < 5:
            return
        
        # We need a PyqtGraph ROI to do the extraction math easily
        # Pos is bottom-left of the unrotated rect.
        # Pivot point is center (0.5, 0.5 relative).
        
        # Center of ROI should be cx, cy.
        # Size (region_w, region_h) corresponds to (Major-Axis-Direction, Minor-Axis-Direction)
        # because we will rotate it by angle_deg.
        
        # Note: pg.ROI expects pos to be the corner.
        # We create a dummy ROI, set its state, and use getArrayRegion
        # We MUST add it to the scene for the transform to work, then remove it.
        
        # 1. Create ROI
        roi = pg.ROI(pos=(cx - region_w/2, cy - region_h/2), 
                     size=(region_w, region_h), 
                     pen=None)
        
        # 2. Setup Transform (Pivot at center, then rotate)
        # Use standard QGraphicsItem methods to rotate around center
        roi.setTransformOriginPoint(region_w / 2, region_h / 2)
        roi.setRotation(angle_deg)
        
        try:
            # 3. Add to scene (Required by getArrayRegion)
            self.vb.addItem(roi)
            
            # 4. Extract
            region_data = roi.getArrayRegion(img, self.imv.getImageItem())
            
        except Exception as e:
            print(f"Profile Extraction FAILED: {e}")
            return
        finally:
            # 5. Cleanup (Remove from scene immediately)
            # This happens so fast the user won't see it (especially in the same event loop tick)
            try:
                self.vb.removeItem(roi)
            except: pass

        if region_data is None or region_data.size == 0:
            return

        # 2. Integrate
        # Major Axis Profile: We walk along the Major Axis (Axis 0) and sum across Minor Axis (Axis 1)
        # This gives Intensity vs Major Axis Position
        prof_major = np.sum(region_data, axis=1)
        
        # Minor Axis Profile: We walk along Minor Axis (Axis 1) and sum across Major Axis (Axis 0)
        prof_minor = np.sum(region_data, axis=0)
        
        # Generate associated spatial axes (centered at 0)
        # Convert to microns
        px_size = self.config.PIXEL_SIZE_UM
        
        x_major = np.linspace(-region_w/2, region_w/2, len(prof_major)) * px_size
        x_minor = np.linspace(-region_h/2, region_h/2, len(prof_minor)) * px_size
        
        # 3. Fits
        # The profile is now an INTEGRAL.
        # Peak of integral ~ Peak_Pixel * Integration_Width roughly.
        # We fit a Gaussian to the integrated profile.
        
        def get_gaussian_fit(x_smooth, x_raw, y_raw, sigma_target):
            if len(y_raw) == 0: return np.zeros_like(x_smooth)
            
            baseline = np.min(y_raw)
            peak = np.max(y_raw) - baseline
            
            # Adjust center (mu) to align fit with the actual data distribution
            # 1st moment of the distribution relative to baseline
            y_clean = y_raw - baseline
            total_intensity = np.sum(y_clean)
            
            if total_intensity > 0:
                mu = np.sum(x_raw * y_clean) / total_intensity
            else:
                mu = 0.0
            
            # Width is fixed based on our D4sigma measurement
            # sigma = d4s / 4  =>  radius = d4s / 2  => 2*sigma^2 = 2*(d4s/4)^2 = d4s^2 / 8
            # In plotting: exp( -2 * (x-mu)^2 / w^2 ) where w is beam radius (D4s/2).
            w_rad = sigma_target / 2.0 
            return baseline + peak * np.exp(-2 * ((x_smooth - mu)**2) / (w_rad**2))

        # Major Axis (Width w)
        # Use high-res X array for smooth fit curve
        x_major_smooth = np.linspace(-region_w/2, region_w/2, 200) * px_size
        fit_major = get_gaussian_fit(x_major_smooth, x_major, prof_major, w * px_size)
        
        self.curve_major_raw.setData(x_major, prof_major)
        self.curve_major_fit.setData(x_major_smooth, fit_major)
        
        # Minor Axis (Width h)
        x_minor_smooth = np.linspace(-region_h/2, region_h/2, 200) * px_size
        fit_minor = get_gaussian_fit(x_minor_smooth, x_minor, prof_minor, h * px_size)
        
        self.curve_minor_raw.setData(x_minor, prof_minor)
        self.curve_minor_fit.setData(x_minor_smooth, fit_minor)
