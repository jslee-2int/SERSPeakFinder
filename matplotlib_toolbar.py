import sys
import matplotlib
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar


class ToolbarWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ToolbarWidget, self).__init__(parent)
        # Create toolbar, passing canvas as the first parameter and parent as the second.

    def read(self, canvas):
        toolbar = NavigationToolbar(canvas, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        # layout.addWidget(canvas)
        self.setLayout(layout)