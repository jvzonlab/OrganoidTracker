import sys
import threading
import tkinter
from os import path
from queue import Queue
from tkinter import StringVar, ttk
from tkinter.font import Font
from typing import List, Dict, Any, Optional

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import Experiment
from gui import dialog
from gui.threading import Scheduler, Task

APP_NAME = "Autotrack"


class Window:
    """The model for a window."""
    __root: tkinter.Tk
    __scheduler: Scheduler

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

        self.__scheduler = Scheduler(root)
        self.__scheduler.daemon = True
        self.__scheduler.start()

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
        """Gets the Matplotlib figure."""
        return self.__fig

    def set_status(self, text: str):
        """Sets the small text below the figure."""
        self.__status_text.set(text)

    def set_figure_title(self, text: str):
        """Sets the big text above the main figure."""
        self.__title_text.set(text)

    def set_window_title(self, text: Optional[str]):
        """Sets the title of the window, prefixed by APP_NAME. Use None as the title to just sown APP_NAME."""
        if text is None:
            self.__root.title(APP_NAME)
        else:
            self.__root.title(APP_NAME + " - " + text)

    def get_experiment(self) -> Experiment:
        """Gets the experiment that is being shown."""
        return self.__experiment

    def set_experiment(self, experiment: Experiment):
        """Replaces the experiment that is being shown. You'll likely want to call refresh() after calling this."""
        self.__experiment = experiment

    def refresh(self):
        """Redraws the main figure."""
        if self.__refresh_handler is not None:
            self.__refresh_handler()

    def setup_menu(self, extra_items: Dict[str, any]):
        """Update the main menu of the window to contain the given options."""
        menu_items = self._get_default_menu()
        self._add_menu_items(menu_items, extra_items)
        self._add_menu_items(menu_items, self._get_default_menu_last())
        _update_menu(self.__menu, menu_items)

    def run_async(self, task: Task):
        """Runs the given task on a worker thread."""
        self.__scheduler.add_task(task)

    def _add_menu_items(self, menu_items, extra_items):
        for name, values in extra_items.items():
            if name in menu_items:
                menu_items[name] = menu_items[name] + values
            else:
                menu_items[name] = values

    def _get_default_menu(self):
        from gui import action

        return {
            "File": [
                ("New project", lambda: action.new(self)),
                "-",
                ("Import images...", lambda: action.load_images(self)),
                ("Import JSON positions and shapes...", lambda: action.load_positions(self)),
                ("Import JSON links, scores and warnings...", lambda: action.load_links(self)),
                ("Import Guizela's track format...", lambda: action.load_guizela_tracks(self)),
                "-",
                ("Export positions and shapes...", lambda: action.export_positions_and_shapes(self.get_experiment())),
                ("Export links...", lambda: action.export_links(self.get_experiment()))
            ],
            "Edit": [],  # This fixes the position of the edit menu
            "View": [
                ("Toggle showing axis numbers", lambda: action.toggle_axis(self.get_figure())),
            ]
        }

    def _get_default_menu_last(self):
        """Some additional options added after all the options from the visualizer are added."""
        from gui import action

        return {
            "File": [
                "-",
                ("Exit (Alt+F4)", lambda: action.ask_exit(self.__root)),
            ],
            "Help": [
                ("Contents...", action.show_manual),
                ("About", action.about_the_program),
            ]
        }

    def has_active_tasks(self) -> bool:
        """Gets whether there are currently tasks being run or scheduled to run."""
        return self.__scheduler.has_active_tasks()


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
            menu.add_command(label=dropdown_item[0], command=_with_safeguard(dropdown_item[1]))
        menu_bar.add_cascade(label=menu_name, menu=menu)


def _with_safeguard(action):
    """Adds an exception handler to the specified lambda action"""
    def safeguard():
        try:
            action()
        except Exception as e:
            dialog.popup_exception(e)
    return safeguard


def launch_window(experiment: Experiment) -> Window:
    """Launches a window with an empty figure. Doesn't start the main loop yet. Use and activate a visualizer to add
    some interactiveness."""
    # Create matplotlib figure
    fig = Figure(figsize=(7, 6), dpi=95)

    # Create empty window
    root = tkinter.Tk()
    root.geometry('800x700')
    root.title(APP_NAME)
    root.iconbitmap(path.join(path.dirname(path.abspath(sys.argv[0])), 'gui', 'icon.ico'))

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
    widget.bind("<Enter>", lambda e: widget.focus_set() if _should_focus(widget) else ...)  # Refocus on mouse enter

    toolbar_frame = ttk.Frame(main_frame)
    toolbar_frame.grid(row=0, column=0, sticky=(tkinter.W, tkinter.E))  # Positions of toolbar buttons
    toolbar = NavigationToolbar2Tk(mpl_canvas, toolbar_frame)
    toolbar.update()

    return window


def _should_focus(widget) -> bool:
    """Returns whether the widget should be focused on mouse hover, which is the case if the current window has
    focus."""
    focus_object = widget.focus_get()
    if focus_object is None:
        return False
    if "toplevel" in str(widget.focus_get()):
        return False
    return True


def mainloop():
    """Starts the main loop."""
    tkinter.mainloop()


