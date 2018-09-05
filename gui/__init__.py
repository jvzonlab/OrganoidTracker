import sys
import tkinter
from os import path
from tkinter import StringVar, ttk
from tkinter.font import Font
from typing import List, Dict, Any, Optional, Iterable, Tuple

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import Experiment
from gui import dialog
from gui.threading import Scheduler, Task

APP_NAME = "Autotrack"


class Plugin:
    """
    Represents a plugin. Plugins can add new data visualizers or provide support for more file types.
    Instead of writing a plugin, you can also write a script that uses the classes in core and imaging.
    """

    def get_menu_items(self, window: "Window")-> Dict[str, Any]:
        """
        Used to add menu items that must always be visible. Example:

            return {
                "File/Import-Import my format...": lambda: my_code_here(),
                "View/Analysis-Useful analysis screen here...": lambda: my_other_code_here()
            }
        """
        return {}

class Window:
    """The model for a window."""
    __root: tkinter.Tk
    __scheduler: Scheduler
    __plugins: List[Plugin]

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
        self.__plugins = []

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
        for plugin in self.__plugins:
            menu_items.update(plugin.get_menu_items(self))
        menu_items.update(extra_items)
        menu_items.update(_get_help_menu())  # This menu must come last
        _update_menu(self.__menu, menu_items)

    def get_scheduler(self) -> Scheduler:
        """Gets the scheduler, useful for registering background tasks"""
        return self.__scheduler

    def install_plugins(self, plugins: Iterable[Plugin]):
        """Adds the given list of plugins to the list of active plugins."""
        for plugin in plugins:
            self.__plugins.append(plugin)

    def _get_default_menu(self) -> Dict[str, Any]:
        from gui import action

        return {
            "File/New-New project...": lambda: action.new(self),
            "File/Import-Import images...": lambda: action.load_images(self),
            "File/Import-Import JSON positions and shapes...": lambda: action.load_positions(self),
            "File/Import-Import JSON links, scores and warnings...": lambda: action.load_links(self),
            "File/Import-Import Guizela's track format...": lambda: action.load_guizela_tracks(self),
            "File/Export-Export positions and shapes...": lambda: action.export_positions_and_shapes(self.get_experiment()),
            "File/Export-Export links...": lambda: action.export_links(self.get_experiment()),
            "File/Exit-Exit (Alt+F4)": lambda: action.ask_exit(self.__root),
            "View/Toggle-Toggle showing axis numbers": lambda: action.toggle_axis(self.get_figure()),
        }


def _simple_menu_dict_to_nested(menu_items: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    menu_tree = {   # Forced order of base menus - these must go first
        "File": {}, "Edit": {}, "View": {}
    }

    for name, action in menu_items.items():
        slash_index = name.index("/")
        main_menu_name, sub_name = name[0:slash_index], name[slash_index + 1:]
        dash_index = sub_name.index("-")
        category_name, label = sub_name[0:dash_index], sub_name[dash_index + 1:]

        if main_menu_name not in menu_tree:
            menu_tree[main_menu_name] = dict()
        categories = menu_tree[main_menu_name]
        if category_name not in categories:
            categories[category_name] = dict()
        category = categories[category_name]
        category[label] = action

    return menu_tree


def _get_help_menu() -> Dict[str, Any]:
    from gui import action

    return {
        "Help/Basic-Contents...": action.show_manual,
        "Help/Basic-About": action.about_the_program,
    }


def _update_menu(menu_bar: tkinter.Menu, menu_items: Dict[str, Any]):
    from gui import action

    menu_tree = _simple_menu_dict_to_nested(menu_items)

    if len(menu_bar.children) > 0:
        menu_bar.delete(0, len(menu_bar.children))  # Remove old menu

    for menu_name, dropdown_items in menu_tree.items():
        # Create each dropdown menu
        if not dropdown_items:
            continue  # Ignore empty menus
        menu = tkinter.Menu(menu_bar, tearoff=0)
        first_category = True
        for category_items in dropdown_items.values():
            if not first_category:
                menu.add_separator()
            else:
                first_category = False

            for item_name, item_action in category_items.items():
                menu.add_command(label=item_name, command=_with_safeguard(item_action))

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


