import os as _os
import re
import subprocess as _subprocess
import sys as _sys
import tkinter as _tkinter
import tkinter.filedialog as _filedialog
import tkinter.messagebox as _messagebox
import tkinter.simpledialog as _simpledialog
import traceback as _traceback
from typing import Tuple, List, Optional

from collections import Callable as _Callable
import matplotlib as _matplotlib
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import UserError


def prompt_int(title: str, question: str) -> Optional[int]:
    """Asks the user to enter an integer. Returns None if the user pressed Cancel or closed the dialog box."""
    return _simpledialog.askinteger(title, question)


def prompt_str(title: str, question: str, default: str = "") -> Optional[str]:
    """Asks the user to enter a line of text. Returns None if the user pressed Cancel or closed the dialog box."""
    return _simpledialog.askstring(title, question, initialvalue=default)


def prompt_confirmation(title: str, question: str):
    """Shows a OK/Cancel dialog box. Returns True if the user pressed OK, returns False if the user pressed Cancel or
    simply closed the dialog box."""
    return _messagebox.askokcancel(title, question)


def prompt_save_file(title: str, file_types: List[Tuple[str, str]]) -> Optional[str]:
    """Shows a prompt that asks the user to save a file. Example:

        prompt_save_file("Save as...", [("PNG file", "*.png"), ("JPEG file", "*.jpg")])

    If the user does not write a file extension, then .png will become the default file extension in this case.
    """
    default_extension = file_types[0][1]
    if re.compile("^\*\.[A-Za-z0-9]+$").match(default_extension):
        default_extension = default_extension[1:]
    else:
        default_extension = None
    return _filedialog.asksaveasfilename(title=title, filetypes=file_types, defaultextension=default_extension)


def prompt_load_file(title: str, file_types: List[Tuple[str, str]]) -> Optional[str]:
    """Shows a prompt that asks the user to open a file. Example:

        prompt_load_file("Choose an image", [("PNG file", "*.png"), ("JPEG file", "*.jpg")])

        Returns None if the user pressed Cancel
    """
    file = _filedialog.askopenfilename(title=title, filetypes=file_types)
    if file == "":
        return None
    return file


def prompt_directory(title: str) -> Optional[str]:
    """Shows a prompt that asks the user to select a directory. Returns None if the user pressed Cancel."""
    directory = _filedialog.askdirectory(title=title)
    if directory == "":
        return None
    return directory


def popup_message_cancellable(title: str, message: str) -> bool:
    """Shows a dialog with Ok and Cancel buttons, but with an "i" sign instead of a "?"."""
    box = _messagebox.Message(title=title, icon=_messagebox.INFO, type=_messagebox.OKCANCEL,message=message).show()
    return str(box) == _messagebox.OK


def popup_error(title: str, message: str):
    _messagebox.showerror(title, message)


def popup_exception(exception: BaseException):
    if isinstance(exception, UserError):
        popup_error(exception.title, exception.body)
        return
    _traceback.print_exception(type(exception), exception, exception.__traceback__)
    popup_error("Internal error", "An error occured.\n" + str(exception) + "\nSee console for technical details.")


def popup_message(title: str, message: str):
    _messagebox.showinfo(title, message)


def popup_figure(draw_function: _Callable):
    """Shows a popup screen with the image"""
    def save_handler(event: KeyEvent):
        if event.key != "ctrl+s":
            return
        file_name = _filedialog.asksaveasfilename(title="Save figure as...", filetypes=(
                ("PNG file", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg")))
        if file_name is None:
            return
        figure.savefig(file_name)

    popup = _tkinter.Toplevel()
    popup.title("Figure")
    popup.grid_columnconfigure(0, weight=1)
    popup.grid_rowconfigure(0, weight=1)

    _matplotlib.rcParams['font.family'] = 'serif'
    _matplotlib.rcParams['font.size'] = 11
    _matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times']

    figure = Figure(figsize=(3.5, 3), dpi=95, tight_layout=True)
    mpl_canvas = FigureCanvasTkAgg(figure, master=popup)
    try:
        draw_function(figure)
    except BaseException as e:
        popup.destroy()
        raise e

    mpl_canvas.mpl_connect("key_press_event", save_handler)
    mpl_canvas.draw()
    mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nesw")


def open_file(filepath: str):
    """Opens a file using the default application."""
    if _sys.platform.startswith('darwin'):
        _subprocess.call(('open', filepath))
    elif _os.name == 'nt':
        _os.startfile(filepath)
    elif _os.name == 'posix':
        _subprocess.call(('xdg-open', filepath))

