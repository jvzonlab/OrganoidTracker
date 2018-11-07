import os as _os
import subprocess as _subprocess
import sys as _sys
import traceback as _traceback
from typing import Tuple, List, Optional

from collections import Callable as _Callable
import matplotlib as _matplotlib
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget, QFileDialog, QInputDialog, QMainWindow
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from autotrack.core import UserError, Name


def _window() -> QWidget:
    return QApplication.activeWindow()


def prompt_int(title: str, question: str) -> Optional[int]:
    """Asks the user to enter an integer. Returns None if the user pressed Cancel or closed the dialog box."""
    result, ok = QInputDialog.getInt(_window(), title, question)
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


class _Popup(QMainWindow):
    _figure: Figure
    _save_name: Name

    def __init__(self, parent: QWidget, figure: Figure, save_name: Name):
        super().__init__(parent)

        self._figure = figure
        figure_widget = FigureCanvasQTAgg(figure)
        figure_widget.setParent(self)
        self.setWindowTitle("Figure")
        self.setCentralWidget(figure_widget)
        figure_widget.mpl_connect("key_press_event", self._save_handler)
        figure_widget.draw()
        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def _save_handler(self, event: KeyEvent):
        if event.key != "ctrl+s":
            return
        file_name = prompt_save_file("Save figure as...", [
            ("PNG file", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg")], suggested_name=self._save_name.get_save_name())
        if file_name is None:
            return
        self._figure.savefig(file_name)


def popup_figure(save_name: Name, draw_function: _Callable):
    """Shows a popup screen with the image"""

    _matplotlib.rcParams['font.family'] = 'serif'
    _matplotlib.rcParams['font.size'] = 11
    _matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times']
    _matplotlib.rcParams['mathtext.fontset'] = 'stix'

    figure = Figure(figsize=(5.5, 5), dpi=95, tight_layout=True)
    try:
        draw_function(figure)
    except BaseException as e:
        raise e
    _Popup(_window(), figure, save_name)


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


