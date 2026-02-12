from PyQt5 import QtWidgets, QtCore
from ORION.src.ui.theme import HEX_ACCENT, HEX_DANGER, HEX_WARNING, HEX_SUCCESS, HEX_TEXT_DIM, HEX_TEXT

class ReadoutWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Readouts", parent)
        self.layout = QtWidgets.QGridLayout(self)
        

        self.lbl_pos = QtWidgets.QLabel("0.000 mm")
        self.lbl_pos.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(QtWidgets.QLabel("Position:"), 0, 0)
        self.layout.addWidget(self.lbl_pos, 0, 1)
        
        self.lbl_d4s = QtWidgets.QLabel("---")
        self.lbl_d4s.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(QtWidgets.QLabel("Beam D4σ:"), 1, 0)
        self.layout.addWidget(self.lbl_d4s, 1, 1)
        
        self.lbl_exposure = QtWidgets.QLabel("1.0 ms")
        self.layout.addWidget(QtWidgets.QLabel("Exposure:"), 2, 0)
        self.layout.addWidget(self.lbl_exposure, 2, 1)

        self.lbl_measure = QtWidgets.QLabel("---")
        self.lbl_measure.setStyleSheet(f"font-size: 12px; color: {HEX_TEXT_DIM};")
        self.layout.addWidget(QtWidgets.QLabel("Last Measurement:"), 3, 0)
        self.layout.addWidget(self.lbl_measure, 3, 1)

        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.layout.addWidget(QtWidgets.QLabel("Status:"), 4, 0)
        self.layout.addWidget(self.lbl_status, 4, 1)

        # Styles removed (handled by global theme)

    def update_stats(self, max_val, d4s, z_pos, exposure):
        self.lbl_pos.setText(f"{z_pos:.3f} mm")
        self.lbl_pos.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {HEX_TEXT};")
        
        if d4s > 0:
            self.lbl_d4s.setText(f"{d4s:.1f} um")
            self.lbl_d4s.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {HEX_ACCENT};")
        else:
            self.lbl_d4s.setText("---")
            self.lbl_d4s.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {HEX_TEXT_DIM};")
            
        self.lbl_exposure.setText(f"{exposure:.2f} ms")
        
        if max_val >= 254:
            self.lbl_exposure.setText(f"{exposure:.2f} ms (SATURATED)")
            self.lbl_exposure.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {HEX_DANGER};")
        else:
            self.lbl_exposure.setStyleSheet(f"font-size: 14px; color: {HEX_TEXT};")

    def update_status(self, msg):
        self.lbl_status.setText(msg)
        m = msg.lower()
        if "error" in m or "fail" in m or "not find" in m:
            self.lbl_status.setStyleSheet(f"color: {HEX_DANGER}; font-weight: bold;")
        elif "searching" in m or "measuring" in m:
            self.lbl_status.setStyleSheet(f"color: {HEX_WARNING}; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet(f"color: {HEX_SUCCESS}; font-weight: bold;")

    def update_measurement(self, msg: str):
        if not msg:
            return
        self.lbl_measure.setText(msg)
        m = msg.lower()
        if "d4s: 0.0" in m or "fail" in m or "not find" in m:
            self.lbl_measure.setStyleSheet(f"font-size: 12px; color: {HEX_WARNING};")
        else:
            self.lbl_measure.setStyleSheet(f"font-size: 12px; color: {HEX_TEXT};")
