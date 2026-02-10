import sys
import argparse
from PyQt5 import QtWidgets
import pyqtgraph as pg

from ORION.config import Config
from ORION.src.drivers.hardware import RealLaserSystem, MockLaserSystem, LaserSystem
from ORION.src.core.worker import HardwareWorker
from ORION.src.ui.layouts.imaging import ImagingPage
from ORION.src.ui.layouts.caustic import CausticPage

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, system: LaserSystem, config: Config):
        super().__init__()
        self.system = system
        self.config = config

        self.worker = HardwareWorker(system, config)
        self.worker.start()

        self.setWindowTitle("ORION: Laser Alignment System")
        self.resize(1000, 800)

        self.init_menu()

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        # Shared system/worker access
        sys_list = [self.system, self.worker]

        self.imaging_page = ImagingPage(sys_list, config)
        self.stack.addWidget(self.imaging_page)

        self.caustic_page = CausticPage(sys_list, config)
        self.stack.addWidget(self.caustic_page)

        self.mems_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.mems_page)
        layout.addWidget(QtWidgets.QLabel("MEMS Control Page - Coming Soon"))
        self.stack.addWidget(self.mems_page)

    def init_menu(self):
        menubar = self.menuBar()

        orion_menu = menubar.addMenu("ORION")
        orion_menu.addAction("New")
        orion_menu.addAction("Reset", self.on_reset)
        orion_menu.addAction("Settings")
        orion_menu.addSeparator()
        orion_menu.addAction("Quit", self.close)
        
        window_menu = menubar.addMenu("Window")
        window_menu.addAction("Imaging", lambda: self.stack.setCurrentIndex(0))
        window_menu.addAction("Caustic", lambda: self.stack.setCurrentIndex(1))
        window_menu.addAction("MEMS", lambda: self.stack.setCurrentIndex(2))
        
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About")

    def on_reset(self):
        QtWidgets.QMessageBox.information(self, "Reset", "System reset triggered.")

    def closeEvent(self, event):
        print("Closing application...")
        self.worker.stop()
        self.system.close()
        event.accept()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()
    
    config = Config()
    
    if args.sim:
        system = MockLaserSystem(config)
    else:
        system = RealLaserSystem(config)
        
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major') 
    
    from ORION.src.ui.theme import apply_theme
    apply_theme(app)
    
    window = MainWindow(system, config)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
