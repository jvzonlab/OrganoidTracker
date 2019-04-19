from typing import Callable

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QToolBar
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from autotrack.gui import dialog
from autotrack.gui.icon_getter import get_icon


class Toolbar(NavigationToolbar2QT):

    toolitems = (  # The icons of Matplotlib that we use. Matplotlib reads these.
        ('Save', 'Save tracking data', 'filesave', 'save_figure'),
        (None, None, None, None),
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    )

    # Handlers for the toolbar buttons
    new_handler = lambda e: ...
    save_handler = lambda e: ...
    load_handler = lambda e: ...
    image_handler= lambda e: ...
    home_handler = lambda e: ...

    def __init__(self, canvas: FigureCanvasQTAgg, parent: QMainWindow):
        super().__init__(canvas, parent)

    def _init_toolbar(self):
        # Add some additional buttons
        self.addAction(get_icon("file_new.png"), "New", lambda *e: self._call(self.new_handler))
        self.addAction(get_icon("file_image.png"), "Load images", lambda *e: self._call(self.image_handler))
        self.addAction(get_icon("file_load.png"), "Load tracking data", lambda *e: self._call(self.load_handler))

        # Add the Matplotlib buttons
        super()._init_toolbar()

    def _call(self, action: Callable):
        try:
            action()
        except BaseException as e:
            dialog.popup_exception(e)

    def home(self):
        self.home_handler()

    def save_figure(self):
        self._call(self.save_handler)
