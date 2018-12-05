import os as _os
import subprocess as _subprocess
import sys as _sys
import traceback as _traceback
from typing import Tuple, List, Optional, Callable

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget, QFileDialog, QInputDialog, QMainWindow, QVBoxLayout, \
    QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from autotrack.core import UserError, Name
from autotrack.gui.window import Window
from autotrack.gui.gui_experiment import GuiExperiment


def _window() -> QWidget:
    active_window = QApplication.activeWindow()
    if active_window is None:
        return QApplication.topLevelWidgets()[0]
    return active_window


def prompt_int(title: str, question: str, min: int=-2147483647, max: int=2147483647) -> Optional[int]:
    """Asks the user to enter an integer. Returns None if the user pressed Cancel or closed the dialog box."""
    result, ok = QInputDialog.getInt(_window(), title, question, min=min, max=max)
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
    if len(file_types) > 1:
        # Create option "All supported file types"
        extensions = set()
        for name, extension in file_types:
            extensions.add(extension)
        file_types = [("All supported file types", ";".join(extensions))] + file_types
    file_types_str = ";;".join((name + "("+extension+ ")" for name, extension in file_types))

    file_name, _ = QFileDialog.getOpenFileName(_window(), title, "", file_types_str)
    if not file_name:
        return None
    return file_name


def prompt_directory(title: str) -> Optional[str]:
    """Shows a prompt that asks the user to select a directory. Returns None if the user pressed Cancel."""
    directory = QFileDialog.getExistingDirectory(_window(), title, "")
    if directory == "":
        return None
    return directory


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

    def __init__(self, parent: QWidget, figure: Figure):
        super().__init__(parent)

        self._figure = figure

        # Initialize main grid
        main_frame = QWidget(parent=self)
        self.setCentralWidget(main_frame)
        vertical_boxes = QVBoxLayout(main_frame)

        # Add title
        self._title_text = QLabel(parent=main_frame)
        self._title_text.setStyleSheet("font-size: 16pt; font-weight: bold")
        vertical_boxes.addWidget(self._title_text)

        # Add Matplotlib figure to frame
        mpl_canvas = FigureCanvasQTAgg(figure)
        mpl_canvas.setParent(main_frame)
        mpl_canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        mpl_canvas.setFocus()
        vertical_boxes.addWidget(mpl_canvas)
        mpl_canvas.draw()

        # Add status bar
        self._status_text = QLabel(parent=main_frame)
        vertical_boxes.addWidget(self._status_text)

        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)


class PopupWindow(Window):
    pass


def popup_window(experiment: GuiExperiment) -> PopupWindow:
    """Pops up a window, which is then returned. You can then for example attach a Visualizer to this window to show
    something.."""
    figure = Figure(figsize=(5.5, 5), dpi=95)
    q_window = _PopupQWindow(_window(), figure)
    return PopupWindow(q_window, figure, experiment, q_window._title_text, q_window._status_text)


def popup_figure(experiment: GuiExperiment, draw_function: Callable[[Figure], None]):
    """Pops up a figure. The figure is drawn inside draw_function."""
    figure = Figure(figsize=(5.5, 5), dpi=95)
    draw_function(figure)
    q_window = _PopupQWindow(_window(), figure)
    PopupWindow(q_window, figure, experiment, q_window._title_text, q_window._status_text)


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
        _os.startfile(filepath)
    elif _os.name == 'posix':
        _subprocess.call(('xdg-open', filepath))


