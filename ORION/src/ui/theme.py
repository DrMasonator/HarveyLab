import sys

from PyQt5 import QtGui, QtWidgets

# Professional Dark Theme Colors (Restored Custom Palette)
COLOR_BG_DARK = "#1e1e1e"        # Main Background
COLOR_BG_LIGHT = "#2b2b2b"       # Secondary Background (Panels, Inputs)
COLOR_BORDER = "#3d3d3d"         # Borders
COLOR_TEXT = "#e0e0e0"           # Main Text
COLOR_TEXT_DIM = "#888888"       # Secondary Text

# Accents
COLOR_ACCENT = "#007acc"         # Blue (Primary)
COLOR_SUCCESS = "#2ea043"        # Green (Success/Go)
COLOR_DANGER = "#da3633"         # Red (Stop/Danger)
COLOR_WARNING = "#bb8800"        # Orange (Warning)

# Exposed as constants for refactored widgets to use if needed
HEX_BG_DARK = COLOR_BG_DARK
HEX_TEXT = COLOR_TEXT
HEX_TEXT_DIM = COLOR_TEXT_DIM

# Convenience hex strings for QSS
HEX_ACCENT = COLOR_ACCENT
HEX_SUCCESS = COLOR_SUCCESS
HEX_DANGER = COLOR_DANGER
HEX_WARNING = COLOR_WARNING

def get_plot_colors():
    """Returns a dict of standard colors for PyQtGraph to match the theme."""
    return {
        'background': COLOR_BG_DARK,
        'axis': COLOR_TEXT,
        'grid': (255, 255, 255, 40), # Low alpha white
        'text': COLOR_TEXT
    }

def apply_theme(app: QtWidgets.QApplication):
    """Applies the global QSS stylesheet to the application."""
    
    # We do NOT use Fusion style here, as user preferred the custom look.

    if sys.platform.startswith("win"):
        font_stack = '"Segoe UI", "Arial", sans-serif'
    elif sys.platform == "darwin":
        font_stack = '"SF Pro Text", "Helvetica Neue", "Arial", sans-serif'
    else:
        font_stack = '"DejaVu Sans", "Liberation Sans", "Arial", sans-serif'

    qss = f"""
    /* --- GLOBAL --- */
    QMainWindow, QWidget {{
        background-color: {COLOR_BG_DARK};
        color: {COLOR_TEXT};
        font-family: {font_stack};
        font-size: 13px;
    }}
    
    QDialog {{
        background-color: {COLOR_BG_DARK};
    }}

    /* --- GROUP BOX --- */
    QGroupBox {{
        background-color: {COLOR_BG_DARK};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        margin-top: 1.2em; /* Leave space for title */
        padding-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        color: {COLOR_TEXT_DIM};
        font-weight: bold;
        background-color: {COLOR_BG_DARK}; /* Mask line behind title */
        padding: 0 3px;
    }}

    /* --- LABELS --- */
    QLabel {{
        color: {COLOR_TEXT};
        border: none;
    }}
    QLabel:disabled {{
        color: {COLOR_TEXT_DIM};
    }}

    /* --- BUTTONS --- */
    QPushButton {{
        background-color: {COLOR_BG_LIGHT};
        border: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT};
        padding: 5px 12px;
        border-radius: 4px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: #3e3e3e;
        border-color: #555;
    }}
    QPushButton:pressed {{
        background-color: #111;
        border-color: #000;
    }}
    QPushButton:disabled {{
        background-color: #1a1a1a;
        color: #555;
        border-color: #333;
    }}

    /* VARIANT: Primary/Accent */
    QPushButton[class="accent"] {{
        background-color: {COLOR_ACCENT};
        border: 1px solid #005a9e;
        color: white;
    }}
    QPushButton[class="accent"]:hover {{
        background-color: #0062a3;
    }}

    /* VARIANT: Success (Green) */
    QPushButton[class="success"] {{
        background-color: {COLOR_SUCCESS};
        border: 1px solid #238636;
        color: white;
    }}
    QPushButton[class="success"]:hover {{
        background-color: #238636;
    }}
    
    /* VARIANT: Danger (Red) */
    QPushButton[class="danger"] {{
        background-color: {COLOR_DANGER};
        border: 1px solid #b62324;
        color: white;
    }}
    QPushButton[class="danger"]:hover {{
        background-color: #b62324;
    }}

    /* --- INPUTS --- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {COLOR_BG_LIGHT};
        border: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT};
        border-radius: 3px;
        padding: 3px;
        selection-background-color: {COLOR_ACCENT};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}
    /* Spinbox Buttons */
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        background-color: transparent;
        border: none;
        width: 16px; 
    }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background-color: #444;
    }}

    /* --- COMBO BOX --- */
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border-left-width: 0px;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLOR_BG_LIGHT};
        border: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT};
    }}

    /* --- TABS --- */
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER};
    }}
    QTabBar::tab {{
        background: {COLOR_BG_LIGHT};
        border: 1px solid {COLOR_BORDER};
        padding: 6px 14px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: {COLOR_TEXT_DIM};
    }}
    QTabBar::tab:selected {{
        background: {COLOR_BG_DARK};
        color: {COLOR_ACCENT};
        border-bottom-color: {COLOR_BG_DARK}; /* Blend with pane */
        font-weight: bold;
    }}
    QTabBar::tab:hover {{
        color: {COLOR_TEXT};
        background: #333;
    }}

    /* --- TABLES --- */
    QTableWidget {{
        background-color: #111;
        gridline-color: {COLOR_BORDER};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER};
    }}
    QHeaderView::section {{
        background-color: {COLOR_BG_LIGHT};
        color: {COLOR_TEXT};
        padding: 4px;
        border: 1px solid {COLOR_BORDER};
    }}
    QTableCornerButton::section {{
        background-color: {COLOR_BG_LIGHT};
        border: 1px solid {COLOR_BORDER};
    }}

    /* --- SCROLLBARS --- */
    QScrollBar:vertical {{
        border: none;
        background: {COLOR_BG_DARK};
        width: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: #444;
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        border: none;
        background: {COLOR_BG_DARK};
        height: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: #444;
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    """
    
    app.setStyleSheet(qss)
