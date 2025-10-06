from typing import Callable, List

from PySide6 import QtGui, QtCore
from PySide6.QtGui import QIcon, QGuiApplication, Qt
from PySide6.QtWidgets import QMainWindow, QComboBox
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from organoid_tracker.gui import dialog, icon_getter


class Toolbar(NavigationToolbar2QT):

    toolitems = [  # The icons of Matplotlib that we use. Matplotlib reads these
        ('Load images', 'Load images', '$custom-icon$file_image', 'button_load_image'),
        ('Load', 'Load tracking data', '$custom-icon$file_load', 'button_load_tracking'),
        ('Save', 'Save tracking data', 'filesave', 'button_save_tracking'),
        (None, None, None, None),
        ('Home', 'Reset original view', 'home', 'button_home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    ]

    # Handlers for the toolbar buttons - set by another class
    new_handler = lambda e: ...
    close_handler = lambda e: ...
    save_handler = lambda e: ...
    load_handler = lambda e: ...
    image_handler= lambda e: ...
    home_handler = lambda e: ...
    next_experiment_handler = lambda e: ...
    previous_experiment_handler = lambda e: ...
    experiment_select_handler = lambda e, experiment_number: ...

    _experiment_selector_box: QComboBox
    _old_experiment_names: List[str] = []

    def __init__(self, canvas: FigureCanvasQTAgg, parent: QMainWindow):
        super().__init__(canvas, parent)
        self.setMovable(False)

        # Set the toolbar style
        dark_mode = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
        background_color = "#191919" if dark_mode else "#FFFFFF"
        border_color = "#3A3A3A" if dark_mode else "#D6D6D6"
        self.setStyleSheet(
            "QToolBar { background: " + background_color + "; padding: 6px; border-top: 1px solid " + border_color + "; border-bottom: 1px solid " + border_color + "; } "
            "QToolBar > QObject { padding: 4px; margin-right: 5px; }")

        # Add experiment selector
        self._experiment_selector_box = QComboBox(self)
        self._experiment_selector_box.currentIndexChanged.connect(
            lambda index: self._call(lambda: self.experiment_select_handler(index) if index != -1 else ...))
        self.update_selectable_experiments([], 0)
        self.addSeparator()
        self.addAction(self._get_our_icon("experiment_previous.png"), "Previous experiment", lambda *e: self._call(self.previous_experiment_handler))
        self.addWidget(self._experiment_selector_box)
        self.addAction(self._get_our_icon("experiment_next.png"), "Next experiment", lambda *e: self._call(self.next_experiment_handler))
        self.addSeparator()
        self.addAction(self._get_our_icon("file_new.png"), "New", lambda *e: self._call(self.new_handler))
        self.addAction(self._get_our_icon("file_close.png"), "Close", lambda *e: self._call(self.close_handler))

    def _icon(self, name):
        # Overridden to provide custom icons (see toolitems at the top of this class)
        if name.startswith("$custom-icon$"):
            # One of our icons!
            name = name[len("$custom-icon$"):]
            return self._get_our_icon(name)
        else:
            # Default matplotlib icon
            return super()._icon(name)

    def _get_our_icon(self, name: str) -> QIcon:
        """Loads an icon, and converts it to dark mode if necessary."""
        # Load the icon
        pixmap = icon_getter.get_icon_pixmap(name)
        pixmap.setDevicePixelRatio(self.devicePixelRatioF() or 1)  # rarely, devicePixelRatioF=0
        # Convert to dark mode if necessary
        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            mask = pixmap.createMaskFromColor(
                QtGui.QColor('black'),
                QtCore.Qt.MaskMode.MaskOutColor)
            pixmap.fill(icon_color)
            pixmap.setMask(mask)
        return QIcon(pixmap)

    def update_selectable_experiments(self, experiment_names: List[str], selected_index: int):
        if len(experiment_names) == 0:
            experiment_names = ["<no data loaded>"]

        # Update the experiment names in the selection box
        if experiment_names != self._old_experiment_names:
            # Don't update GUI if not necessary

            self._old_experiment_names = experiment_names
            # Change names of existing items, add new items if necessary
            for i, experiment_name in enumerate(experiment_names):
                if i >= self._experiment_selector_box.count():
                    self._experiment_selector_box.addItem(experiment_name)
                else:
                    self._experiment_selector_box.setItemText(i, experiment_name)
            # Remove superfluous items
            for i in range(len(experiment_names), self._experiment_selector_box.count()):
                self._experiment_selector_box.removeItem(self._experiment_selector_box.count() - 1)

        if self._experiment_selector_box.currentIndex() != selected_index:
            self._experiment_selector_box.setCurrentIndex(selected_index)

    def _call(self, action: Callable):
        try:
            action()
        except BaseException as e:
            dialog.popup_exception(e)

    def button_load_image(self):
        self._call(self.image_handler)

    def button_load_tracking(self):
        self._call(self.load_handler)

    def button_save_tracking(self):
        self._call(self.save_handler)

    def button_home(self):
        self._call(self.home_handler)
