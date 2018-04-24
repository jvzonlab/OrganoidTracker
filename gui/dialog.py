import tkinter
from tkinter import simpledialog, messagebox
from tkinter.messagebox import Message

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def prompt_int(title: str, question: str) -> int:
    return simpledialog.askinteger(title, question)


def message_cancellable(title: str, message: str) -> bool:
    """Shows a dialog with Ok and Cancel buttons, but with an "i" sign instead of a "?"."""
    box = Message(title=title, icon=messagebox.INFO, type=messagebox.OKCANCEL,message=message).show()
    return str(box) == messagebox.OK


def popup_error(title: str, message: str):
    messagebox.showerror(title, message)


def popup_figure(draw_function):
    """Shows a popup screen with the image"""
    popup = tkinter.Toplevel()
    popup.title("Figure")
    popup.grid_columnconfigure(0, weight=1)
    popup.grid_rowconfigure(0, weight=1)

    figure = Figure(figsize=(7, 7), dpi=95)
    draw_function(figure)

    mpl_canvas = FigureCanvasTkAgg(figure, master=popup)
    mpl_canvas.draw()
    mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nesw")