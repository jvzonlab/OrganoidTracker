import json
import os as _os
import subprocess as _subprocess
import sys as _sys
import traceback as _traceback
from enum import Enum
from statistics import median
from typing import Tuple, List, Optional, Callable, Any, Dict, Union

from PySide6 import QtCore
from PySide6.QtGui import QCloseEvent, QColor
from PySide6.QtWidgets import QMessageBox, QApplication, QWidget, QFileDialog, QInputDialog, QMainWindow, QVBoxLayout, \
    QLabel, QSizePolicy, QColorDialog, QMenuBar
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from organoid_tracker.core import UserError, Name, CM_TO_INCH, Color
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.text_popup.text_popup import RichTextPopup
from organoid_tracker.gui.window import Window


def _window() -> QWidget:
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QMainWindow):
            return widget
    return QApplication.topLevelWidgets()[0]


def prompt_int(title: str, question: str, *, minimum: int = -2147483647, maximum: int = 2147483647,
               default=0) -> Optional[int]:
    """Asks the user to enter an integer. Returns None if the user pressed Cancel or closed the dialog box. If the
    minimum is equal to the maximum, that number is returned. The default value is automatically clamped to the minimum
    and maximum."""
    if minimum == maximum:
        return minimum
    default = median([minimum, default, maximum])  # Make the default bounded by the minimum and maximum
    result, ok = QInputDialog.getInt(_window(), title, question, minValue=minimum, maxValue=maximum, value=default)
    return result if ok else None


def prompt_float(title: str, question: str, minimum: float = -1.0e10, maximum: float = 1.0e10, default: float = 0,
                 decimals: int = 1) -> Optional[float]:
    """Asks the user to enter a float. Returns None if the user pressed Cancel or closed the dialog box. The default
    value is automatically clamped to the minimum and maximum."""
    default = median([minimum, default, maximum])  # Make the default bounded by the minimum and maximum
    result, ok = QInputDialog.getDouble(_window(), title, question, minValue=minimum, maxValue=maximum, value=default,
                                        decimals=decimals)
    return result if ok else None


def prompt_str(title: str, question: str, default: str = "") -> Optional[str]:
    """Asks the user to enter a line of text. Returns None if the user pressed Cancel or closed the dialog box."""
    text, ok = QInputDialog.getText(_window(), title, question, text=default)
    return text if ok else None


def prompt_confirmation(title: str, question: str):
    """Shows a OK/Cancel dialog box. Returns True if the user pressed OK, returns False if the user pressed Cancel or
    simply closed the dialog box."""
    result = QMessageBox.question(_window(), title, question, QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
    return result == QMessageBox.Ok


class DefaultOption(Enum):
    OK = "OK"
    CANCEL = "CANCEL"


def prompt_options(title: str, question: str, *, option_1: str, option_2: Optional[str] = None,
                   option_3: Optional[str] = None, option_default: DefaultOption = DefaultOption.CANCEL
                   ) -> Optional[int]:
    """Shows two or three options to choose from, and additionally a cancel button. Returns None if the default button
    (either Cancel or Ok) was pressed, returns a number (1, 2 or 3, depending on the picked option) otherwise."""

    # Set up window
    button_1, button_2, button_3 = None, None, None
    box = QMessageBox(_window())
    box.setWindowTitle(title)
    box.setText(question)
    button_1 = box.addButton(option_1, QMessageBox.ActionRole)
    if option_2 is not None:
        button_2 = box.addButton(option_2, QMessageBox.ActionRole)
        if option_3 is not None:
            button_3 = box.addButton(option_3, QMessageBox.ActionRole)
    if option_default == DefaultOption.OK:
        box.addButton(QMessageBox.Ok)
    else:
        box.addButton(QMessageBox.Cancel)

    result = box.exec_()
    if result == QMessageBox.Cancel or result == QMessageBox.Ok:
        # Clicked one of these standard buttons
        return None
    clicked_button = box.clickedButton()
    if clicked_button == button_1:
        return 1
    if button_2 is not None and clicked_button == button_2:
        return 2
    if button_3 is not None and clicked_button == button_3:
        return 3
    return None  # Don't know what was clicked


def prompt_save_file(title: str, file_types: List[Tuple[str, str]], suggested_name: Optional[Name] = None
                     ) -> Optional[str]:
    """Shows a prompt that asks the user to save a file. Example:

        prompt_save_file("Save as...", [("PNG file", "*.png"), ("JPEG file", "*.jpg")])

    If the user does not write a file extension, then the file extension will be added automatically.
    """
    file_types_str = ";;".join((name + "(" + extension + ")" for name, extension in file_types))

    file_name, _ = QFileDialog.getSaveFileName(_window(), title, "", file_types_str)
    if not file_name:
        return None
    return file_name


def prompt_load_file(title: str, file_types: List[Tuple[str, str]]) -> Optional[str]:
    """Shows a prompt that asks the user to open a file. Example:

        prompt_load_file("Choose an image", [("PNG file", "*.png"), ("JPEG file", "*.jpg")])

    Returns None if the user pressed Cancel. This function automatically adds an "All supported files" option.
    """
    file_name, _ = QFileDialog.getOpenFileName(_window(), title, "", _to_file_types_str(file_types))
    if not file_name:
        return None
    return file_name


def prompt_load_multiple_files(title: str, file_types: List[Tuple[str, str]]) -> List[str]:
    """Shows a prompt that asks the user to open multiple files. Example:

        prompt_load_files("Choose images", [("PNG file", "*.png"), ("JPEG file", "*.jpg")])

    Returns an empty list if the user pressed Cancel. This function automatically adds an "All supported files" option.
    """
    file_names, _ = QFileDialog.getOpenFileNames(_window(), title, "", _to_file_types_str(file_types))
    if not file_names:
        return []
    return file_names


def _to_file_types_str(file_types: List[Tuple[str, str]]) -> str:
    if len(file_types) > 1:
        # Create option "All supported file types"
        extensions = set()
        for name, extension in file_types:
            extensions.add(extension)
        file_types = [("All supported file types", ";".join(extensions))] + file_types
    return ";;".join((name + "("+extension+ ")" for name, extension in file_types))


def prompt_directory(title: str) -> Optional[str]:
    """Shows a prompt that asks the user to select a directory. Returns None if the user pressed Cancel."""
    directory = QFileDialog.getExistingDirectory(_window(), title, "")
    if directory == "":
        return None
    return directory


def prompt_color(title: str, default_color: Color = Color(255, 255, 255)) -> Optional[Color]:
    """Prompts the user to choose a color."""
    q_color = QColor(default_color.red, default_color.green, default_color.blue)
    q_color = QColorDialog.getColor(q_color, _window(), title)
    if q_color.isValid():
        return Color(q_color.red(), q_color.green(), q_color.blue())
    return None


def popup_message_cancellable(title: str, message: str) -> bool:
    """Shows a dialog with Ok and Cancel buttons, but with an "i" sign instead of a "?"."""
    result = QMessageBox.information(_window(), title, message, QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
    return result == QMessageBox.Ok


def popup_error(title: str, message: str):
    QMessageBox.critical(_window(), title, message, QMessageBox.Ok, QMessageBox.Ok)


def popup_exception(exception: BaseException):
    if isinstance(exception, UserError):
        popup_error(exception.title, exception.body)
        return
    _traceback.print_exception(type(exception), exception, exception.__traceback__)
    popup_error("Internal error", "An error occured.\n" + str(exception) + "\nSee console for technical details.")


def popup_message(title: str, message: str):
    QMessageBox.information(_window(), title, message, QMessageBox.Ok, QMessageBox.Ok)


class _PopupQWindow(QMainWindow):

    _figure: Figure
    _status_text: QLabel
    _title_text: QLabel
    _menu: QMenuBar
    _on_close: Callable

    def __init__(self, parent: QWidget, figure: Figure, draw_function: Callable, on_close: Callable):
        super().__init__(parent)

        self._figure = figure
        self._on_close = on_close

        # Initialize main grid
        main_frame = QWidget(parent=self)
        self.setCentralWidget(main_frame)
        vertical_boxes = QVBoxLayout(main_frame)

        # Add title
        self._title_text = QLabel(parent=main_frame)
        self._title_text.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        vertical_boxes.addWidget(self._title_text)

        # Add Matplotlib figure to frame
        mpl_canvas = FigureCanvasQTAgg(figure)
        mpl_canvas.setParent(main_frame)
        mpl_canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        mpl_canvas.setFocus()
        vertical_boxes.addWidget(mpl_canvas)
        figure.set_canvas(mpl_canvas)
        draw_function(figure)
        mpl_canvas.draw()

        # Add Matplotlib toolbar
        self.addToolBar(NavigationToolbar2QT(mpl_canvas, self))

        # Add status bar
        self._status_text = QLabel(parent=main_frame)
        self._status_text.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed))
        vertical_boxes.addWidget(self._status_text)

        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def closeEvent(self, event: QCloseEvent):
        try:
            self._on_close()
        except BaseException as e:
            popup_exception(e)



def _as_gui_experiment(object: Union[GuiExperiment, Window]) -> GuiExperiment:
    """Helper method so that you can pass both a Window and a GuiExperiment instance, but always get a GuiExperiment
    out."""
    if isinstance(object, GuiExperiment):
        return object
    return object.get_gui_experiment()


class PopupWindow(Window):

    def __init__(self, q_window: QMainWindow, figure: Figure, parent_window: Union[GuiExperiment, Window],
                 title_text: QLabel, status_text: QLabel):
        # parent_window used to always be a GuiExperiment. Now we also support passing a window object instead
        # (hence the name change), so that we can access the plugin manager too.
        super().__init__(q_window, figure, _as_gui_experiment(parent_window), title_text, status_text)
        if isinstance(parent_window, Window):
            self.replace_plugin_manager(parent_window.plugin_manager)

    def _get_default_menu(self) -> Dict[str, Any]:
        return {
            "File//Save-Save figure...": self._save_figure
        }

    def _save_figure(self):
        figure = self.get_figure()
        file_types = list()
        for abbreviation, information in figure.canvas.get_supported_filetypes().items():
            file_types.append((information, "*." + abbreviation))
        file_name = prompt_save_file("Save figure as", file_types)
        if file_name is not None:
            self.get_figure().savefig(file_name)


def popup_visualizer(parent_window: Window, visualizer_callable: Callable[[Window], Any], *,
                     size_cm: Tuple[float, float] = (14, 12.7)):
    """Pops up a window, which is then returned. You can then for example attach a Visualizer to this window to show
    something. The size is the size of the figure, not the window. The window will be slightly larger than the figure.
    """
    figure = Figure(figsize=(size_cm[0] * CM_TO_INCH, size_cm[1] * CM_TO_INCH), dpi=95)

    def close_listener():
        visualizer.detach()
    def no_draw(_: Figure):
        pass
    q_window = _PopupQWindow(_window(), figure, no_draw, close_listener)
    window = PopupWindow(q_window, figure, parent_window, q_window._title_text, q_window._status_text)

    from organoid_tracker.visualizer import Visualizer
    visualizer: Visualizer = visualizer_callable(window)
    menu = visualizer.get_extra_menu_options()
    if len(menu) > 0:
        window.setup_menu(menu, show_plugins=False)
    visualizer.attach()
    visualizer.refresh_all()
    visualizer.update_status(visualizer.get_default_status())


def popup_figure(parent_window: Union[GuiExperiment, Window], draw_function: Callable[[Figure], None], *,
                 size_cm: Tuple[float, float] = (14, 12.7),
                 export_function: Optional[Callable[[], Dict[str, Any]]] = None):
    """Pops up a figure. The figure is drawn inside draw_function. Size (x, y) is specified using size_cm.
    export_function is used to save a JSON structure when the user presses Ctrl+E.

    For backwards compatibility, parent_window can still be a GuiExperiment. However, then plugins cannot
    access anything in the popup window.
    """
    def do_nothing_on_close():
        pass  # Used to indicate that no action needs to be taken once the window closes

    figure = Figure(figsize=(size_cm[0] * CM_TO_INCH, size_cm[1] * CM_TO_INCH), dpi=95)
    q_window = _PopupQWindow(_window(), figure, draw_function, do_nothing_on_close)

    # Support for exporting the raw data
    def try_export(event: KeyEvent):
        if event.key != "ctrl+e":
            return
        if export_function is None:
            popup_error("Export not available", "Export to raw data is not available for this figure.")
            return
        file = prompt_save_file("Raw data", [("JSON files", "*.json")])
        if file is None:
            return
        data = export_function()
        with open(file, 'w') as handle:
            json.dump(data, handle)
    figure.canvas.mpl_connect("key_release_event", try_export)

    PopupWindow(q_window, figure, parent_window, q_window._title_text, q_window._status_text)


def prompt_yes_no(title: str, message: str) -> bool:
    """Asks a Yes/No question. Returns True if the user pressed Yes and False otherwise."""
    result = QMessageBox.question(_window(), title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return result == QMessageBox.Yes


class PromptAnswer:
    _answer: int

    def __init__(self, answer: int):
        self._answer = answer

    def is_yes(self):
        return self._answer == QMessageBox.Yes

    def is_no(self):
        return self._answer == QMessageBox.No

    def is_cancel(self):
        return self._answer == QMessageBox.Cancel


def prompt_yes_no_cancel(title: str, message: str) -> PromptAnswer:
    """Asks a yes/no/cancel question."""
    result = QMessageBox.question(_window(), title, message, QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                  QMessageBox.Cancel)
    return PromptAnswer(result)


def open_file(filepath: str):
    """Opens a file using the default application."""
    if _sys.platform.startswith('darwin'):
        _subprocess.call(('open', filepath))
    elif _os.name == 'nt':
        _os.startfile(filepath.replace("/", "\\"))  # Windows requires backslashes for network paths like \\server\bla
    elif _os.name == 'posix':
        _subprocess.call(('xdg-open', filepath))


def popup_manual():
    """Shows the manual of the program."""
    from organoid_tracker.gui import _rich_text_dialog
    from organoid_tracker.text_popup import help_popup
    _rich_text_dialog.show_popup(_window(), help_popup.HelpPopup())


def popup_rich_text(text: RichTextPopup):
    """Pops up a rich text window, so that you can easily show dialogs with text, links and images."""
    from organoid_tracker.gui import _rich_text_dialog
    _rich_text_dialog.show_popup(_window(), text)
