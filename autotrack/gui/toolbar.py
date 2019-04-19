from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QToolBar
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from autotrack.gui import dialog


class Toolbar(NavigationToolbar2QT):

    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

    # Called when the Save button is pressed
    save_handler = lambda e: ...
    # Called when the Home button is pressed
    home_handler = lambda e: ...

    def __init__(self, canvas: FigureCanvasQTAgg, parent: QMainWindow):
        super().__init__(canvas, parent)

    def home(self):
        self.home_handler()

    def save_figure(self):
        try:
            self.save_handler()
        except BaseException as e:
            dialog.popup_exception(e)
