import tkinter
from tkinter import StringVar, ttk
from tkinter.font import Font

from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg, FigureCanvasTkAgg
from matplotlib.figure import Figure


class Window:
    """The model for a window."""
    __fig: Figure
    __status_text: StringVar
    __title_text: StringVar

    def __init__(self, figure: Figure, title_text: StringVar, status_text: StringVar):
        self.__fig = figure
        self.__status_text = status_text
        self.__title_text = title_text

    def register_event_handler(self, event: str, action):
        self.__fig.canvas.mpl_connect(event, action)

    def get_figure(self) -> Figure:
        return self.__fig

    def set_status(self, text: str):
        self.__status_text.set(text)

    def set_title(self, text: str):
        self.__title_text.set(text)


def _create_menu(root: tkinter.Tk, window: Window):
    menu_bar = tkinter.Menu(root)

    file_menu = tkinter.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Exit", command=lambda: _action_exit(root))

    view_menu = tkinter.Menu(menu_bar, tearoff=0)
    view_menu.add_command(label="Toggle axis", command=lambda: _action_toggle_axis(window.get_figure()))

    menu_bar.add_cascade(label="File", menu=file_menu)
    menu_bar.add_cascade(label="View", menu=view_menu)
    return menu_bar


def _action_exit(root: tkinter.Tk):
    root.quit()
    root.destroy()

def _action_toggle_axis(figure: Figure):
    set_visible = None
    for axis in figure.axes:
        if set_visible is None:
            set_visible = not axis.get_xaxis().get_visible()
        axis.get_xaxis().set_visible(set_visible)
        axis.get_yaxis().set_visible(set_visible)
    figure.canvas.draw()


def launch_window() -> Window:
    """Launches a window with an empty figure. Doesn't start the main loop yet."""
    # Create matplotlib figure
    fig = Figure(figsize=(7, 6), dpi=95)

    # Create empty window
    root = tkinter.Tk()
    title_text = StringVar()
    status_text = StringVar()
    window = Window(fig, title_text, status_text)
    root.title("Autotrack")
    root.iconbitmap('gui/icon.ico')
    root.geometry('800x700')
    root.config(menu=_create_menu(root, window))
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(1, weight=1)

    # Initialize main grid
    main_frame = ttk.Frame(root)
    main_frame.grid(column=1, row=1, sticky="nesw")
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(2, weight=1)

    # Add title and status bar
    ttk.Label(main_frame, textvariable=title_text,
              font=Font(size=14, weight="bold"),
              padding=(0, 10, 0, 10)).grid(row=1, column=0, sticky="we")
    ttk.Label(main_frame, textvariable=status_text).grid(row=3, column=0, sticky="wes")

    # Add Matplotlib figure to frame
    mpl_canvas = FigureCanvasTkAgg(fig, master=main_frame)  # A tk.DrawingArea.
    mpl_canvas.show()
    mpl_canvas.get_tk_widget().grid(row=2, column=0, sticky="we")  # Position of figure

    toolbar_frame = ttk.Frame(main_frame)
    toolbar_frame.grid(row=0, column=0, sticky=(tkinter.W, tkinter.E))  # Positions of toolbar buttons
    toolbar = NavigationToolbar2TkAgg(mpl_canvas, toolbar_frame)
    toolbar.update()

    return window


def mainloop():
    """Starts the main loop."""
    tkinter.mainloop()


def popup_figure(draw_function):
    """Shows a popup screen with the image"""
    popup = tkinter.Toplevel()
    popup.title("Test")

    figure = Figure(figsize=(4, 4), dpi=95)
    draw_function(figure)

    mpl_canvas = FigureCanvasTkAgg(figure, master=popup)
    mpl_canvas.show()
    mpl_canvas.get_tk_widget().pack()