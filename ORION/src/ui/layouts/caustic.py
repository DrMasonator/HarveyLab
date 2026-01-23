import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from ORION.config import Config
from ORION.src.drivers.hardware import LaserSystem
from ORION.src.core.worker import HardwareWorker
from ORION.src.core.analysis import BeamAnalyzer
from ORION.src.core.worker import WorkerMode
from ORION.src.ui.theme import HEX_BG_DARK, HEX_SUCCESS

class CausticPage(QtWidgets.QWidget):
    def __init__(self, systemList: list, config: Config):
        super().__init__()
        self.system = systemList[0]
        self.worker = systemList[1]
        self.config = config
        self.analyzer = BeamAnalyzer(config)
        
        self.data_points = [] # List of dict: {'z': float, 'dx': float, 'dy': float, 'deff': float}
        self.fit_results_x = {}
        self.fit_results_y = {}
        
        # State
        self.current_z = 0.0
        self.expecting_measurement = False
        
        self.init_ui()
        self.init_connections()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Plot Area
        self.plot_container = pg.GraphicsLayoutWidget()
        self.plot_container.setBackground(HEX_BG_DARK)
        
        self.plot = self.plot_container.addPlot(title="Caustic Measurement")
        self.plot.setLabel('left', "Beam Diameter", units='um')
        self.plot.setLabel('bottom', "Z Position", units='mm')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        
        # Curves
        self.curve_eff_raw = self.plot.plot(pen=None, symbol='t', symbolBrush='#00ff00', symbolPen='#00ff00', name='Eff (Geo Mean)')
        self.curve_x_raw = self.plot.plot(pen=None, symbol='o', symbolBrush='#00aaff', symbolPen='#00aaff', name='X Raw')
        self.curve_y_raw = self.plot.plot(pen=None, symbol='s', symbolBrush='#ffaa00', symbolPen='#ffaa00', name='Y Raw')
        
        self.curve_eff_fit = self.plot.plot(pen=pg.mkPen('#00ff00', width=2, style=QtCore.Qt.DashLine), name='Eff Fit')
        self.curve_x_fit = self.plot.plot(pen=pg.mkPen('#00aaff', width=2), name='X Fit')
        self.curve_y_fit = self.plot.plot(pen=pg.mkPen('#ffaa00', width=2), name='Y Fit')
        
        # Current Position Line
        self.line_current_z = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1, style=QtCore.Qt.DashLine))
        self.plot.addItem(self.line_current_z)
        
        layout.addWidget(self.plot_container, stretch=2)
        
        # Bottom Controls
        bottom_widget = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0,0,0,0)
        
        # Controls
        ctrl_group = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_group)
        
        self.btn_measure = QtWidgets.QPushButton("Add Point (Measure)")
        self.btn_measure.setFixedHeight(40)
        self.btn_measure.setProperty("class", "success")
        
        self.btn_clear = QtWidgets.QPushButton("Clear All Points")
        
        ctrl_layout.addWidget(self.btn_measure)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addStretch()
        
        # Scan Settings
        scan_group = QtWidgets.QGroupBox("Scan")
        scan_layout = QtWidgets.QFormLayout(scan_group)
        
        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_start.setRange(0, 500)
        self.spin_start.setValue(0.0)
        self.spin_start.setSuffix(" mm")
        
        self.spin_end = QtWidgets.QDoubleSpinBox()
        self.spin_end.setRange(0, 500)
        self.spin_end.setValue(10.0)
        self.spin_end.setSuffix(" mm")
        
        self.spin_step = QtWidgets.QDoubleSpinBox()
        self.spin_step.setDecimals(4)
        self.spin_step.setRange(self.config.SEARCH_TOLERANCE, 10.0)
        self.spin_step.setValue(0.1)
        self.spin_step.setSingleStep(0.01)
        self.spin_step.setSuffix(" mm")
        
        self.btn_start_scan = QtWidgets.QPushButton("Start Scan")
        
        scan_layout.addRow("Start Z:", self.spin_start)
        scan_layout.addRow("End Z:", self.spin_end)
        scan_layout.addRow("Step:", self.spin_step)
        
        hbox_scan = QtWidgets.QHBoxLayout()
        self.btn_start_scan = QtWidgets.QPushButton("Manual Scan")
        
        hbox_scan.addWidget(self.btn_start_scan)
        
        scan_layout.addRow(hbox_scan)
        
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setProperty("class", "danger")
        scan_layout.addRow(self.btn_stop)
        
        ctrl_layout.addWidget(scan_group)

        # Manual Move
        move_group = QtWidgets.QGroupBox("Move Z")
        move_layout = QtWidgets.QHBoxLayout(move_group)
        
        self.spin_move_z = QtWidgets.QDoubleSpinBox()
        self.spin_move_z.setRange(0, 500)
        self.spin_move_z.setSuffix(" mm")
        self.spin_move_z.setDecimals(3)
        
        self.btn_go_z = QtWidgets.QPushButton("Go")
        
        move_layout.addWidget(self.spin_move_z)
        move_layout.addWidget(self.btn_go_z)
        
        ctrl_layout.addWidget(move_group)
        
        # Data Table
        table_group = QtWidgets.QGroupBox("Measurements")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Z [mm]", "Dx [mm]", "Dy [mm]"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table_layout.addWidget(self.table)
        
        # Results
        res_group = QtWidgets.QGroupBox("Results (M² Fit)")
        res_layout = QtWidgets.QGridLayout(res_group)
        
        # Labels
        res_layout.addWidget(QtWidgets.QLabel("Parameter"), 0, 0)
        res_layout.addWidget(QtWidgets.QLabel("X-Axis"), 0, 1)
        res_layout.addWidget(QtWidgets.QLabel("Y-Axis"), 0, 2)
        
        # Params
        self.lbl_m2_x = QtWidgets.QLabel("---")
        self.lbl_m2_y = QtWidgets.QLabel("---")
        self.add_res_row(res_layout, 1, "M²", self.lbl_m2_x, self.lbl_m2_y)
        
        self.lbl_d0_x = QtWidgets.QLabel("---")
        self.lbl_d0_y = QtWidgets.QLabel("---")
        self.add_res_row(res_layout, 2, "Waist d0 [um]", self.lbl_d0_x, self.lbl_d0_y)
        
        self.lbl_z0_x = QtWidgets.QLabel("---")
        self.lbl_z0_y = QtWidgets.QLabel("---")
        self.add_res_row(res_layout, 3, "Waist Z [mm]", self.lbl_z0_x, self.lbl_z0_y)
        
        self.lbl_zr_x = QtWidgets.QLabel("---")
        self.lbl_zr_y = QtWidgets.QLabel("---")
        self.add_res_row(res_layout, 4, "Rayleigh Zr [mm]", self.lbl_zr_x, self.lbl_zr_y)
        
        self.lbl_div_x = QtWidgets.QLabel("---")
        self.lbl_div_y = QtWidgets.QLabel("---")
        self.add_res_row(res_layout, 5, "Divergence [mrad]", self.lbl_div_x, self.lbl_div_y)

        # Style labels
        for lbl in [self.lbl_m2_x, self.lbl_m2_y, self.lbl_d0_x, self.lbl_d0_y, 
                    self.lbl_z0_x, self.lbl_z0_y, self.lbl_zr_x, self.lbl_zr_y,
                    self.lbl_div_x, self.lbl_div_y]:
             lbl.setStyleSheet(f"color: {HEX_SUCCESS}; font-weight: bold;")


        bottom_layout.addWidget(ctrl_group, stretch=1)
        bottom_layout.addWidget(table_group, stretch=2)
        bottom_layout.addWidget(res_group, stretch=2)
        
        layout.addWidget(bottom_widget, stretch=1)
        
    def add_res_row(self, layout, row, name, lbl_x, lbl_y):
        layout.addWidget(QtWidgets.QLabel(name), row, 0)
        layout.addWidget(lbl_x, row, 1)
        layout.addWidget(lbl_y, row, 2)

    def init_connections(self):
        self.btn_measure.clicked.connect(self.on_measure)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_start_scan.clicked.connect(self.on_manual_scan)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_go_z.clicked.connect(self.on_go_z)
        
        self.worker.stats_update.connect(self.on_stats_update)
        
        self.worker.stats_update.connect(self.on_stats_update)
        
        self.worker.overlay_update.connect(self.handle_overlay_update)
        self.worker.one_shot_finished.connect(self.handle_oneshot_finished)
        self.worker.measurement_taken.connect(self.handle_measurement_taken)
        
    def on_stats_update(self, max_val, d4s, z_pos, exposure):
        self.current_z = z_pos
        self.line_current_z.setPos(z_pos)
        
    def on_measure(self):
        self.expecting_measurement = True
        self.btn_measure.setEnabled(False)
        self.btn_measure.setText("Measuring...")
        self.worker.start_one_shot_measurement()
        
    def on_manual_scan(self):
        start_z = self.spin_start.value()
        end_z = self.spin_end.value()
        step = self.spin_step.value()
        
        # Check against global soft limit
        soft_limit = self.worker.system.soft_limit
        if end_z > soft_limit:
            QtWidgets.QMessageBox.warning(self, "Limit Reached", f"End Z ({end_z}) exceeds Safe Limit ({soft_limit}). Clamped.")
            end_z = soft_limit
        if start_z > soft_limit:
            QtWidgets.QMessageBox.warning(self, "Limit Reached", f"Start Z ({start_z}) exceeds Safe Limit ({soft_limit}). Clamped.")
            start_z = soft_limit
            
        # Determine direction
        if start_z > end_z and step > 0:
            step = -step
        
        # Generate linear points
        if abs(step) < 1e-9: return # Avoid div by zero
        num_steps = int(abs((end_z - start_z) / step)) + 1
        points = list(np.linspace(start_z, end_z, num_steps))
        
        # Double check points individually just in case float math pushed them over
        points = [p for p in points if p <= soft_limit + 1e-6] 
        
        self.worker.start_characterization(points)
        QtWidgets.QMessageBox.information(self, "Scan Started", f"Manual scan running ({len(points)} points)...")
    def on_stop(self):
        self.worker.request_stop()
        
    def on_go_z(self):
        val = self.spin_move_z.value()
        self.worker.set_target_z(val)

    def handle_measurement_taken(self, z, d_eff, d_x, d_y):
        if self.worker.mode == WorkerMode.SEARCH:
            return
            
            
        # robust_measure_optical returns d4s in MICRONS, convert to mm for UI
        dx_mm = d_x / 1000.0
        dy_mm = d_y / 1000.0
        
        self.add_point(z, dx_mm, dy_mm)
        
    def handle_overlay_update(self, cx, cy, wpx, hpx, phi):
        # Convert measurement to mm
        pix_um = self.config.PIXEL_SIZE_UM
        dx_mm = (wpx * pix_um) / 1000.0
        dy_mm = (hpx * pix_um) / 1000.0
        
        if getattr(self, '_pending_smart_scan', False):
            self._pending_smart_scan = False
            # Use geometric mean or max as d0? standard is d4s effective
            d0_est = np.sqrt(dx_mm * dy_mm)
            self.plan_smart_scan(d0_est)
            return

        if not self.expecting_measurement:
            return
            
        self.expecting_measurement = False
        self.btn_measure.setEnabled(True)
        self.btn_measure.setText("Add Point (Measure)")

        z_mm = self.current_z
        
        self.add_point(z_mm, dx_mm, dy_mm)
        
    def handle_oneshot_finished(self, msg):
        # This signal comes at the end, whether success or fail.
        # If we failed (D4s: 0.0), handle_overlay_update was NOT called.
        if self.expecting_measurement:
            # Check if failure
            if "D4s: 0.0" in msg:
                self.expecting_measurement = False
                self.btn_measure.setEnabled(True)
                self.btn_measure.setText("Add Point (Measure)")
                QtWidgets.QMessageBox.warning(self, "Measurement Failed", "Could not find beam. Check settings.")

    def on_clear(self):
        self.data_points = []
        self.table.setRowCount(0)
        self.fit_results_x = {}
        self.fit_results_y = {}
        self.replot()
        self.update_results_display()

    def add_point(self, z, d_x, d_y):
        d_eff = np.sqrt(d_x * d_y)
        self.data_points.append({'z': z, 'dx': d_x, 'dy': d_y, 'deff': d_eff})
        self.data_points.sort(key=lambda p: p['z'])
        
        self.update_table()
        
        self.recalculate_fit()
        
    def update_table(self):
        self.table.setRowCount(0)
        for p in self.data_points:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{p['z']:.3f}"))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{p['dx']:.3f}"))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{p['dy']:.3f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{p['deff']:.3f}"))
        
    def recalculate_fit(self):
        if len(self.data_points) < 3:
            return 
            
        zs = [p['z'] for p in self.data_points]
        dxs = [p['dx'] for p in self.data_points]
        dys = [p['dy'] for p in self.data_points]
        deffs = [p['deff'] for p in self.data_points]
        
        res_x = self.analyzer.calculate_caustic_fit(zs, dxs, self.config.WAVELENGTH_NM)
        res_y = self.analyzer.calculate_caustic_fit(zs, dys, self.config.WAVELENGTH_NM)
        
        # Effective Fit (Geometric Mean of fits if possible)
        res_eff = {}
        if res_x.get('success') and res_y.get('success'):
            # Generate common Z based on fit ranges
            zx = res_x['fit_z']
            zy = res_y['fit_z']
            z_min = min(zx[0], zy[0])
            z_max = max(zx[-1], zy[-1])
            
            fit_z = np.linspace(z_min, z_max, 200)
            
            # Eval fits on common Z
            def eval_hyperbola(params, z_arr):
                d0 = params['d0_mm']
                z0 = params['z0_mm']
                theta = params['theta_mrad'] / 1000.0
                return np.sqrt(d0**2 + (theta**2) * (z_arr - z0)**2)

            fit_d_x_new = eval_hyperbola(res_x, fit_z)
            fit_d_y_new = eval_hyperbola(res_y, fit_z)
            
            # Calc mean
            fit_d_eff = np.sqrt(fit_d_x_new * fit_d_y_new)
            
            res_eff['success'] = True
            res_eff['fit_z'] = fit_z
            res_eff['fit_d'] = fit_d_eff
            
            # Derive mean params
            res_eff['M2'] = np.sqrt(res_x['M2'] * res_y['M2'])
            res_eff['d0'] = np.sqrt(res_x['d0'] * res_y['d0'])
            res_eff['z0'] = (res_x['z0'] + res_y['z0']) / 2 
            res_eff['zr'] = np.sqrt(res_x['zr'] * res_y['zr']) 
            res_eff['div'] = np.sqrt(res_x['div'] * res_y['div'])
        else:
             # Fallback
             res_eff = self.analyzer.calculate_caustic_fit(zs, deffs, self.config.WAVELENGTH_NM)
        
        self.fit_results_x = res_x
        self.fit_results_y = res_y
        self.fit_results_eff = res_eff
        
        self.replot()
        self.update_results_display()

    def replot(self):
        zs = [p['z'] for p in self.data_points]
        
        if zs:
            # Convert mm -> um for plotting
            self.curve_x_raw.setData(zs, [p['dx']*1000.0 for p in self.data_points])
            self.curve_y_raw.setData(zs, [p['dy']*1000.0 for p in self.data_points])
            self.curve_eff_raw.setData(zs, [p['deff']*1000.0 for p in self.data_points])
        else:
            self.curve_x_raw.clear()
            self.curve_y_raw.clear()
            self.curve_eff_raw.clear()
        
        if 'fit_z' in self.fit_results_x:
            self.curve_x_fit.setData(self.fit_results_x['fit_z'], self.fit_results_x['fit_d'] * 1000.0)
        else:
            self.curve_x_fit.clear()
            
        if 'fit_z' in self.fit_results_y:
            self.curve_y_fit.setData(self.fit_results_y['fit_z'], self.fit_results_y['fit_d'] * 1000.0)
        else:
            self.curve_y_fit.clear()

        if 'fit_z' in getattr(self, 'fit_results_eff', {}):
             self.curve_eff_fit.setData(self.fit_results_eff['fit_z'], self.fit_results_eff['fit_d'] * 1000.0)
        else:
             self.curve_eff_fit.clear()

    def update_results_display(self):
        def fmt(val, prec=2):
            if val is None: return "---"
            if isinstance(val, (int, float)):
                return f"{val:.{prec}f}"
            return str(val)
            
        # X
        rx = self.fit_results_x
        if 'M2' in rx:
            self.lbl_m2_x.setText(fmt(rx.get('M2'), 3))   # 3 decimals for M2
            # d0 comes in mm. User wants um.
            d0_mm = rx.get('d0_mm', 0)
            self.lbl_d0_x.setText(fmt(d0_mm * 1000.0, 2)) # 2 dec for um (e.g. 10.50 um)
            self.lbl_z0_x.setText(fmt(rx.get('z0_mm'), 4))# 4 dec for Z mm (precision important)
            self.lbl_zr_x.setText(fmt(rx.get('zR_mm'), 4))# 4 dec for Zr mm
            self.lbl_div_x.setText(fmt(rx.get('theta_mrad'), 3)) # 3 dec for Div
        else:
            # Clear invalid
            for l in [self.lbl_m2_x, self.lbl_d0_x, self.lbl_z0_x, self.lbl_zr_x, self.lbl_div_x]:
                l.setText("---")

        # Y
        ry = self.fit_results_y
        if 'M2' in ry:
            self.lbl_m2_y.setText(fmt(ry.get('M2'), 3))
            d0_mm = ry.get('d0_mm', 0)
            self.lbl_d0_y.setText(fmt(d0_mm * 1000.0, 2))
            self.lbl_z0_y.setText(fmt(ry.get('z0_mm'), 4))
            self.lbl_zr_y.setText(fmt(ry.get('zR_mm'), 4))
            self.lbl_div_y.setText(fmt(ry.get('theta_mrad'), 3))
        else:
             for l in [self.lbl_m2_y, self.lbl_d0_y, self.lbl_z0_y, self.lbl_zr_y, self.lbl_div_y]:
                l.setText("---")
