from PyQt5 import QtWidgets, QtCore

from ORION.config import Config


class SettingsPage(QtWidgets.QWidget):
    settings_applied = QtCore.pyqtSignal()
    back_requested = QtCore.pyqtSignal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.fields = {}

        self._build_ui()
        self.load_from_config(self.config)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        self.btn_back = QtWidgets.QPushButton("Back to Imaging")
        self.btn_back.clicked.connect(lambda: self.back_requested.emit())
        header.addWidget(self.btn_back)
        layout.addLayout(header)

        self.lbl_path = QtWidgets.QLabel(f"Config file: {self.config.default_path()}")
        self.lbl_path.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self.lbl_path)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        container = QtWidgets.QWidget()
        scroll.setWidget(container)
        form_layout = QtWidgets.QVBoxLayout(container)
        form_layout.setSpacing(12)

        form_layout.addWidget(self._build_exposure_group())
        form_layout.addWidget(self._build_detection_group())
        form_layout.addWidget(self._build_roi_group())
        form_layout.addWidget(self._build_search_group())
        form_layout.addStretch()

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_save = QtWidgets.QPushButton("Save Settings")
        self.btn_save.setProperty("class", "accent")
        self.btn_reset.clicked.connect(self.on_reset_defaults)
        self.btn_save.clicked.connect(self.on_save)
        buttons.addWidget(self.btn_reset)
        buttons.addWidget(self.btn_save)
        layout.addLayout(buttons)

    def _build_exposure_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Exposure & Auto-Exposure")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self._add_float(
            layout,
            "Start Exposure",
            "START_EXPOSURE_MS",
            0.001,
            5000.0,
            0.1,
            3,
            " ms",
            "Initial exposure used at startup before auto-exposure stabilizes.",
        )
        self._add_float(
            layout,
            "Min Exposure",
            "MIN_EXPOSURE_MS",
            0.001,
            5000.0,
            0.1,
            3,
            " ms",
            "Lower limit for auto-exposure.",
        )
        self._add_float(
            layout,
            "Max Exposure",
            "MAX_EXPOSURE_MS",
            0.001,
            10000.0,
            1.0,
            3,
            " ms",
            "Upper limit for auto-exposure.",
        )
        self._add_int(
            layout,
            "Target Brightness Min",
            "TARGET_BRIGHTNESS_MIN",
            0,
            255,
            1,
            "",
            "Auto-exposure tries to keep peak intensity above this.",
        )
        self._add_int(
            layout,
            "Target Brightness Max",
            "TARGET_BRIGHTNESS_MAX",
            0,
            255,
            1,
            "",
            "Auto-exposure tries to keep peak intensity below this.",
        )
        self._add_int(
            layout,
            "Absolute Saturation",
            "ABSOLUTE_SATURATION",
            0,
            255,
            1,
            "",
            "If the max pixel exceeds this, exposure is reduced immediately.",
        )
        self._add_int(
            layout,
            "Low Signal Threshold",
            "LOW_SIGNAL_THRESHOLD",
            0,
            255,
            1,
            "",
            "Below this peak level, the beam is treated as low-signal.",
        )
        self._add_bool(
            layout,
            "Find Beam Skip AE",
            "FIND_BEAM_SKIP_AE",
            "Skip auto-exposure during Find Beam for speed.",
        )
        self._add_int(
            layout,
            "Find Beam Average Count",
            "FIND_BEAM_AVERAGE_COUNT",
            1,
            50,
            1,
            "",
            "Number of frames averaged during Find Beam.",
        )
        self._add_int(
            layout,
            "Measure Average Count",
            "MEASURE_AVERAGE_COUNT",
            1,
            100,
            1,
            "",
            "Number of frames averaged for normal D4s measurements.",
        )
        self._add_choice(
            layout,
            "Bayer Mode",
            "BAYER_MODE",
            ["RAW", "RED", "GREEN", "BLUE"],
            "Which color plane to analyze (RAW uses the full sensor).",
        )

        return group

    def _build_detection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Detection")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self._add_float(
            layout,
            "Noise Cutoff Percent",
            "NOISE_CUTOFF_PERCENT",
            0.01,
            0.8,
            0.01,
            3,
            "",
            "Pixels below this fraction of peak are ignored during detection.",
        )
        return group

    def _build_roi_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("ROI Tracking")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self._add_int(
            layout,
            "ROI Min Size",
            "ROI_MIN_SIZE_PX",
            50,
            5000,
            10,
            " px",
            "Minimum ROI size for tracking the beam.",
        )
        self._add_int(
            layout,
            "ROI Max Size",
            "ROI_MAX_SIZE_PX",
            100,
            10000,
            10,
            " px",
            "Maximum ROI size to prevent huge processing areas.",
        )
        self._add_int(
            layout,
            "ROI Search Block",
            "ROI_SEARCH_BLOCK",
            4,
            128,
            1,
            " px",
            "Block size used for coarse peak search.",
        )
        self._add_float(
            layout,
            "ROI Edge Peak Fraction",
            "ROI_EDGE_PEAK_FRACTION",
            0.001,
            0.2,
            0.001,
            3,
            "",
            "If ROI border has this fraction of peak, ROI expands.",
        )
        self._add_float(
            layout,
            "ROI Expand Threshold",
            "ROI_EXPAND_THRESHOLD",
            0.1,
            0.9,
            0.01,
            3,
            "",
            "Expand ROI if beam span exceeds this fraction of ROI.",
        )
        self._add_float(
            layout,
            "ROI Shrink Threshold",
            "ROI_SHRINK_THRESHOLD",
            0.05,
            0.7,
            0.01,
            3,
            "",
            "Shrink ROI if beam span is below this fraction of ROI.",
        )
        self._add_int(
            layout,
            "ROI Hysteresis Frames",
            "ROI_ADAPT_HYSTERESIS_FRAMES",
            1,
            20,
            1,
            "",
            "Number of frames before ROI expands/shrinks.",
        )
        return group

    def _build_search_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Search & Focus")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self._add_float(
            layout,
            "Search Tolerance",
            "SEARCH_TOLERANCE",
            0.0001,
            0.05,
            0.0001,
            4,
            " mm",
            "Stop autofocus when the bracket is within this range.",
        )
        return group

    def _add_float(self, layout, label, key, min_v, max_v, step, decimals, suffix, help_text=""):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(min_v, max_v)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(suffix)
        layout.addRow(label + ":", self._wrap_with_help(w, help_text))
        self.fields[key] = w

    def _add_int(self, layout, label, key, min_v, max_v, step, suffix="", help_text=""):
        w = QtWidgets.QSpinBox()
        w.setRange(min_v, max_v)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(suffix)
        layout.addRow(label + ":", self._wrap_with_help(w, help_text))
        self.fields[key] = w

    def _add_bool(self, layout, label, key, help_text=""):
        w = QtWidgets.QCheckBox()
        layout.addRow(label + ":", self._wrap_with_help(w, help_text))
        self.fields[key] = w

    def _add_choice(self, layout, label, key, options, help_text=""):
        w = QtWidgets.QComboBox()
        w.addItems(options)
        w.setMinimumWidth(140)
        layout.addRow(label + ":", self._wrap_with_help(w, help_text))
        self.fields[key] = w

    def _wrap_with_help(self, widget, help_text: str):
        if not help_text:
            return widget
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget)
        info = QtWidgets.QLabel("ℹ")
        info.setToolTip(help_text)
        info.setStyleSheet("color: #9aa0a6; font-size: 12px; padding-left: 6px;")
        row.addWidget(info)
        row.addStretch()
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(row)
        return wrapper

    def load_from_config(self, config: Config):
        for key, widget in self.fields.items():
            val = getattr(config, key)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QtWidgets.QComboBox):
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            else:
                widget.setValue(val)

    def on_reset_defaults(self):
        defaults = Config()
        self.load_from_config(defaults)

    def _normalize(self):
        if self.config.MIN_EXPOSURE_MS > self.config.MAX_EXPOSURE_MS:
            self.config.MIN_EXPOSURE_MS, self.config.MAX_EXPOSURE_MS = (
                self.config.MAX_EXPOSURE_MS,
                self.config.MIN_EXPOSURE_MS,
            )
        if self.config.TARGET_BRIGHTNESS_MIN > self.config.TARGET_BRIGHTNESS_MAX:
            self.config.TARGET_BRIGHTNESS_MIN, self.config.TARGET_BRIGHTNESS_MAX = (
                self.config.TARGET_BRIGHTNESS_MAX,
                self.config.TARGET_BRIGHTNESS_MIN,
            )
        if self.config.ROI_MIN_SIZE_PX > self.config.ROI_MAX_SIZE_PX:
            self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX = (
                self.config.ROI_MAX_SIZE_PX,
                self.config.ROI_MIN_SIZE_PX,
            )

    def on_save(self):
        for key, widget in self.fields.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                val = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                val = widget.currentText()
            else:
                val = widget.value()
            setattr(self.config, key, val)

        self._normalize()
        self.config.save()
        self.load_from_config(self.config)
        self.settings_applied.emit()
