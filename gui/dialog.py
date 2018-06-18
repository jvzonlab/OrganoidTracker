import os
import subprocess
import sys
import tkinter
import traceback
import logging
from collections import Callable
from tkinter import simpledialog, messagebox, filedialog
from tkinter.messagebox import Message

import matplotlib
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import UserError


def prompt_int(title: str, question: str) -> int:
    return simpledialog.askinteger(title, question)


def message_cancellable(title: str, message: str) -> bool:
    """Shows a dialog with Ok and Cancel buttons, but with an "i" sign instead of a "?"."""
    box = Message(title=title, icon=messagebox.INFO, type=messagebox.OKCANCEL,message=message).show()
    return str(box) == messagebox.OK


def popup_error(title: str, message: str):
    messagebox.showerror(title, message)


def popup_exception(exception: BaseException):
    if isinstance(exception, UserError):
        popup_error(exception.title, exception.body)
        return
    traceback.print_exception(type(exception), exception, exception.__traceback__)
    popup_error("Internal error", "An error occured.\n" + str(exception) + "\nSee console for technical details.")


def popup_message(title: str, message: str):
    messagebox.showinfo(title, message)


def popup_figure(draw_function: Callable):
    """Shows a popup screen with the image"""
    def save_handler(event: KeyEvent):
        if event.key != "ctrl+s":
            return
        file_name = filedialog.asksaveasfilename(title="Save figure as...", filetypes=(
                ("PNG file", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg")))
        if file_name is None:
            return
        figure.savefig(file_name)

    popup = tkinter.Toplevel()
    popup.title("Figure")
    popup.grid_columnconfigure(0, weight=1)
    popup.grid_rowconfigure(0, weight=1)

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times']

    figure = Figure(figsize=(7, 4.32), dpi=95)
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
    if sys.platform.startswith('darwin'):
        subprocess.call(('open', filepath))
    elif os.name == 'nt':
        os.startfile(filepath)
    elif os.name == 'posix':
        subprocess.call(('xdg-open', filepath))
