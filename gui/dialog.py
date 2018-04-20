import tkinter
from tkinter import simpledialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def prompt_int(title: str, question: str) -> int:
    return simpledialog.askinteger(title, question)


def popup_figure(draw_function):
    """Shows a popup screen with the image"""
    popup = tkinter.Toplevel()
    popup.title("Test")

    figure = Figure(figsize=(4, 4), dpi=95)
    draw_function(figure)

    mpl_canvas = FigureCanvasTkAgg(figure, master=popup)
    mpl_canvas.show()
    mpl_canvas.get_tk_widget().pack()