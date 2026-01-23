from PyQt5 import QtWidgets, QtCore, QtGui
from ORION.src.ui.theme import HEX_DANGER

class MotorControlWidget(QtWidgets.QGroupBox):
    # Signals for parent to handle
    move_requested = QtCore.pyqtSignal(float)
    find_beam_requested = QtCore.pyqtSignal()
    find_focus_requested = QtCore.pyqtSignal()
    measure_requested = QtCore.pyqtSignal()
    step_requested = QtCore.pyqtSignal(float) # -1.0 or 1.0 direction
    set_safe_requested = QtCore.pyqtSignal()
    reset_safe_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.layout = QtWidgets.QGridLayout(self)
        
        # Row 1: Go to Z & Find Beam
        self.layout.addWidget(QtWidgets.QLabel("Go to Z:"), 0, 0)
        self.txt_move_z = QtWidgets.QLineEdit()
        self.txt_move_z.setFixedWidth(60)
        self.txt_move_z.setValidator(QtGui.QDoubleValidator())
        self.txt_move_z.returnPressed.connect(self.emit_move_requested)
        self.layout.addWidget(self.txt_move_z, 0, 1)
        
        self.btn_find_beam = QtWidgets.QPushButton("Find beam")
        self.btn_find_beam.setProperty("class", "accent")
        self.btn_find_beam.clicked.connect(lambda: self.find_beam_requested.emit())
        self.layout.addWidget(self.btn_find_beam, 0, 2)

        self.btn_find_focus = QtWidgets.QPushButton("Find Focus")
        self.btn_find_focus.setProperty("class", "accent")
        self.btn_find_focus.clicked.connect(lambda: self.find_focus_requested.emit())
        self.layout.addWidget(self.btn_find_focus, 0, 3)
        
        # Row 2: Z-step & Measure
        self.layout.addWidget(QtWidgets.QLabel("Z-step:"), 1, 0)
        self.combo_step = QtWidgets.QComboBox()
        self.combo_step.addItems(["10um", "100um", "1mm"])
        self.combo_step.setFixedWidth(90)
        self.layout.addWidget(self.combo_step, 1, 1)
        
        self.btn_measure = QtWidgets.QPushButton("Measure D4σ")
        self.btn_measure.setProperty("class", "accent")
        self.btn_measure.clicked.connect(lambda: self.measure_requested.emit())
        self.layout.addWidget(self.btn_measure, 1, 2)
        
        self.check_overlay = QtWidgets.QCheckBox("Indicator") # Added label
        self.check_overlay.setChecked(True)
        self.layout.addWidget(self.check_overlay, 1, 3)
        
        # Row 3: Jog & Safe Z
        jog_layout = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("<")
        self.btn_back.setFixedWidth(35)
        self.btn_back.clicked.connect(lambda: self.step_requested.emit(-1.0))
        
        self.btn_fwd = QtWidgets.QPushButton(">")
        self.btn_fwd.setFixedWidth(35)
        self.btn_fwd.clicked.connect(lambda: self.step_requested.emit(1.0))
        
        jog_layout.addWidget(self.btn_back)
        jog_layout.addWidget(self.btn_fwd)
        jog_layout.addStretch()
        self.layout.addLayout(jog_layout, 2, 0, 1, 2)
        
        self.btn_set_safe = QtWidgets.QPushButton("Set safe Z")
        self.btn_set_safe.clicked.connect(lambda: self.set_safe_requested.emit())
        self.layout.addWidget(self.btn_set_safe, 2, 2)
        
        # Safe Z display and Reset button layout
        safe_z_layout = QtWidgets.QHBoxLayout()
        self.lbl_safe_val = QtWidgets.QLabel("---")
        self.lbl_safe_val.setStyleSheet(f"font-weight: bold; color: {HEX_DANGER};")
        
        self.btn_reset_safe = QtWidgets.QPushButton("X")
        self.btn_reset_safe.setFixedWidth(20)
        self.btn_reset_safe.setToolTip("Reset Safe Limit")
        self.btn_reset_safe.clicked.connect(lambda: self.reset_safe_requested.emit())
        # Inline for transparency
        self.btn_reset_safe.setStyleSheet(f"""
            QPushButton {{ padding: 0px; color: {HEX_DANGER}; font-weight: bold; border: none; background: transparent; }}
            QPushButton:hover {{ color: red; }}
        """)
        
        safe_z_layout.addWidget(self.lbl_safe_val)
        safe_z_layout.addWidget(self.btn_reset_safe)
        
        self.layout.addLayout(safe_z_layout, 2, 3)

        # Premium Common Styling
        # Premium Common Styling - Removed, handled by global theme
        # Specific IDs for styling - Removed


    def set_safe_z_display(self, val):
        self.lbl_safe_val.setText(f"{val:.2f}mm")

    def emit_move_requested(self):
        txt = self.txt_move_z.text()
        if txt:
            try:
                val = float(txt)
                self.move_requested.emit(val)
                # Clear focus so the user knows it's submitted
                self.txt_move_z.clearFocus()
            except ValueError:
                pass
