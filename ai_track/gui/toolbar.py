from typing import Callable, List

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QToolBar, QComboBox
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from ai_track.gui import dialog
from ai_track.gui.icon_getter import get_icon


class Toolbar(NavigationToolbar2QT):

    toolitems = (  # The icons of Matplotlib that we use. Matplotlib reads these.
        ('Save', 'Save tracking data', 'filesave', 'save_figure'),
        (None, None, None, None),
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    )

    # Handlers for the toolbar buttons - set by another class
    new_handler = lambda e: ...
    close_handler = lambda e: ...
    save_handler = lambda e: ...
    load_handler = lambda e: ...
    image_handler= lambda e: ...
    home_handler = lambda e: ...
    experiment_select_handler = lambda e, experiment_number: ...

    _experiment_selector_box: QComboBox
    _old_experiment_names: List[str] = []

    def __init__(self, canvas: FigureCanvasQTAgg, parent: QMainWindow):
        super().__init__(canvas, parent)

        # Add experiment selector
        self._experiment_selector_box = QComboBox(self)
        self._experiment_selector_box.currentIndexChanged.connect(
            lambda index: self._call(lambda: self.experiment_select_handler(index) if index != -1 else ...))
        self.update_selectable_experiments([], 0)
        self.addSeparator()
        self.addWidget(self._experiment_selector_box)
        self.addAction(get_icon("file_new.png"), "New", lambda *e: self._call(self.new_handler))
        self.addAction(get_icon("file_close.png"), "Close", lambda *e: self._call(self.close_handler))

    def update_selectable_experiments(self, experiment_names: List[str], selected_index: int):
        if len(experiment_names) == 0:
            experiment_names = ["<no data loaded>"]

        # Don't update GUI if not necessary
        if experiment_names != self._old_experiment_names:
            self._old_experiment_names = experiment_names

            self._experiment_selector_box.clear()
            for experiment_name in experiment_names:
                self._experiment_selector_box.addItem(experiment_name)

        self._experiment_selector_box.setCurrentIndex(selected_index)

    def _init_toolbar(self):
        # Add some additional buttons
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

