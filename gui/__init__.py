import tkinter
from tkinter import StringVar, ttk
from tkinter.font import Font
from typing import List, Dict, Any

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import Experiment


class Window:
    """The model for a window."""
    __root: tkinter.Tk

    __fig: Figure
    __status_text: StringVar
    __title_text: StringVar
    __experiment: Experiment

    __event_handler_ids: List[int]
    __refresh_handler: Any = None
    __menu: tkinter.Menu

    def __init__(self, root: tkinter.Tk, menu: tkinter.Menu, figure: Figure, experiment: Experiment,
                 title_text: StringVar, status_text: StringVar):
        self.__root = root
        self.__menu = menu
        self.__fig = figure
        self.__experiment = experiment
        self.__status_text = status_text
        self.__title_text = title_text
        self.__event_handler_ids = []

    def register_event_handler(self, event: str, action):
        """Registers an event handler. Supported events:

        * All matplotlib events.
        * "refresh_event" for when the figure needs to be redrawn.
        """
        if event == "refresh_event":
            self.__refresh_handler = action

        self.__event_handler_ids.append(self.__fig.canvas.mpl_connect(event, action))

    def unregister_event_handlers(self):
        """Unregisters all handles registered using register_event_handler"""
        for id in self.__event_handler_ids:
            self.__fig.canvas.mpl_disconnect(id)
        self.__event_handler_ids = []
        self.__refresh_handler = None
        self.setup_menu(dict())

    def get_figure(self) -> Figure:
        return self.__fig

    def set_status(self, text: str):
        self.__status_text.set(text)

    def set_title(self, text: str):
        self.__title_text.set(text)

    def get_experiment(self) -> Experiment:
        return self.__experiment

    def set_experiment(self, experiment: Experiment):
        self.__experiment = experiment

    def refresh(self):
        """Redraws the main figure."""
        if self.__refresh_handler is not None:
            self.__refresh_handler()

    def setup_menu(self, extra_items: Dict[str, any]):
        menu_items = self._get_default_menu()
        for name, values in extra_items.items():
            if name in menu_items:
                menu_items[name] = menu_items[name] + values
            else:
                menu_items[name] = values
        _update_menu(self.__menu, menu_items)

    def _get_default_menu(self):
        from gui import action

        return {
            "File": [
                ("New project", lambda: action.new(self)),
                "-",
                ("Import images...", lambda: action.load_images(self)),
                ("Import JSON positions...", lambda: action.load_positions(self)),
                ("Import JSON links...", lambda: action.load_links(self)),
                ("Import Guizela's track format...", lambda: action.load_guizela_tracks(self)),
                "-",
                ("Export positions and shapes...", lambda: action.export_positions_and_shapes(self.get_experiment())),
                ("Export links...", lambda: action.export_links(self.get_experiment())),
                "-",
                ("Exit (Alt+F4)", lambda: action.ask_exit(self.__root)),
            ],
            "Edit": [],  # This fixes the position of the edit menu
            "View": [
                ("Toggle axis numbers", lambda: action.toggle_axis(self.get_figure())),
            ]
        }


def _update_menu(menu_bar: tkinter.Menu, menu_items: Dict[str, any]):
    from gui import action
    if len(menu_bar.children) > 0:
        menu_bar.delete(0, len(menu_bar.children))  # Remove old menu
    for menu_name, dropdown_items in menu_items.items():
        if len(dropdown_items) == 0:
            continue
        menu = tkinter.Menu(menu_bar, tearoff=0)
        for dropdown_item in dropdown_items:
            if dropdown_item == "-":
                menu.add_separator()
                continue
            menu.add_command(label=dropdown_item[0], command=dropdown_item[1])
        menu_bar.add_cascade(label=menu_name, menu=menu)


def launch_window(experiment: Experiment) -> Window:
    """Launches a window with an empty figure. Doesn't start the main loop yet. Use and activate a visualizer to add
    some interactiveness."""
    # Create matplotlib figure
    fig = Figure(figsize=(7, 6), dpi=95)

    # Create empty window
    root = tkinter.Tk()
    root.geometry('800x700')
    root.title("Autotrack")
    root.iconbitmap('gui/icon.ico')

    title_text = StringVar()
    status_text = StringVar()
    menu = tkinter.Menu()
    window = Window(root, menu, fig, experiment, title_text, status_text)

    window.setup_menu(dict())  # This draws the menu
    root.config(menu=menu)

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
    ttk.Label(main_frame, textvariable=status_text).grid(row=3, column=0, sticky="we")

    # Add Matplotlib figure to frame
    mpl_canvas = FigureCanvasTkAgg(fig, master=main_frame)  # A tk.DrawingArea.
    mpl_canvas.draw()
    widget = mpl_canvas.get_tk_widget()
    widget.grid(row=2, column=0, sticky="we")  # Position of figure
    widget.bind("<Enter>", lambda e: widget.focus_set())  # Refocus on mouse enter

    toolbar_frame = ttk.Frame(main_frame)
    toolbar_frame.grid(row=0, column=0, sticky=(tkinter.W, tkinter.E))  # Positions of toolbar buttons
    toolbar = NavigationToolbar2Tk(mpl_canvas, toolbar_frame)
    toolbar.update()

    return window


def mainloop():
    """Starts the main loop."""
    tkinter.mainloop()


