# -*- coding: utf-8 -*-
#
# Copyright © 2009 Pierre Raybaut
# Licensed under the terms of the MIT License

"""
MatplotlibWidget
================
Example of matplotlib widget for PyQt4
Copyright © 2009 Pierre Raybaut
This software is licensed under the terms of the MIT License
Derived from 'embedding_in_pyqt4.py':
Copyright © 2005 Florent Rougon, 2006 Darren Dale
"""

__version__ = "1.0.0"

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import QSize
from qbstyles import mpl_style

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['font.size'] = 5


class MatplotlibWidget(Canvas):
    """
    MatplotlibWidget with subplot functionality.
    """
    def __init__(self, parent=None, title='', xlabel='', ylabel='',
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 width=4, height=3, dpi=75, hold=False, bg_color='#F9F9F9',
                 rows=1, cols=1, index=1):
        mpl_style(dark=False)
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor=bg_color)

        # Subplot grid setup
        self.rows = rows
        self.cols = cols
        self.axes = self.figure.add_subplot(rows, cols, index)

        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if xscale is not None:
            self.axes.set_xscale(xscale)
        if yscale is not None:
            self.axes.set_yscale(yscale)
        if xlim is not None:
            self.axes.set_xlim(*xlim)
        if ylim is not None:
            self.axes.set_ylim(*ylim)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)
        Canvas.updateGeometry(self)

    def set_subplot(self, index):
        """
        Change to a specific subplot in the grid.
        """
        self.axes = self.figure.add_subplot(self.rows, self.cols, index)

    def clear_all(self):
        """
        Clear the entire figure including all subplots.
        """
        self.figure.clear()  # Clear the entire figure
        self.axes = None  # Reset the axes attribute
        self.draw()  # Update the canvas

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)


# ===============================================================================
# Example with Subplots
# ===============================================================================
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QMainWindow, QApplication
    from numpy import linspace


    class ApplicationWindow(QMainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            # Create MatplotlibWidget with a 2x1 grid
            self.mplwidget = MatplotlibWidget(self, title='Example',
                                              xlabel='Linear scale',
                                              ylabel='Log scale',
                                              rows=2, cols=1, hold=True)
            self.mplwidget.setFocus()
            self.setCentralWidget(self.mplwidget)

            # Plot on the first subplot
            self.mplwidget.set_subplot(1)
            self.plot(self.mplwidget.axes, x_power=2)

            # Plot on the second subplot
            self.mplwidget.set_subplot(2)
            self.plot(self.mplwidget.axes, x_power=3)

        def plot(self, axes, x_power):
            x = linspace(-10, 10)
            axes.tick_params(axis="y", direction="in", pad=10)
            axes.tick_params(axis="x", direction="in", pad=10)
            axes.plot(x, x ** x_power, label=f'x^{x_power}')
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=1)


    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    app.exec()
