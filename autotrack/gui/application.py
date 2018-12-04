from typing import List, Iterable, Dict, Any

from PyQt5.QtWidgets import QWidget

from autotrack.gui.threading import Scheduler


class Plugin:
    """
    Represents a plugin. Plugins can add new data visualizers or provide support for more file types.
    Instead of writing a plugin, you can also write a script that uses the classes in core and imaging.
    """

    def get_menu_items(self, window: "Window") -> Dict[str, Any]:
        """
        Used to add menu items that must always be visible. Example:

            return {
                "File/Import-Import my format...": lambda: my_code_here(),
                "View/Analysis-Useful analysis screen here...": lambda: my_other_code_here()
            }
        """
        return {}

    def reload(self):
        """Reloads this plugin from disk."""
        ...


class Application:
    __scheduler: Scheduler
    __plugins: List[Plugin]

    def __init__(self, scheduler_widget: QWidget):
        self.__plugins = list()

        self.__scheduler = Scheduler(scheduler_widget)
        self.__scheduler.daemon = True
        self.__scheduler.start()

    @property
    def scheduler(self):
        """Gets the scheduler, used to run tasks on another thread."""
        return self.__scheduler

    def install_plugins(self, plugins: Iterable[Plugin]):
        """Adds the given list of plugins to the list of active plugins."""
        for plugin in plugins:
            self.__plugins.append(plugin)

    def reload_plugins(self) -> int:
        """Reloads all plugins from disk. You should update the window after calling this. Returns the number of
        reloaded plugins."""
        for plugin in self.__plugins:
            plugin.reload()
        return len(self.__plugins)

    def get_plugins(self) -> Iterable[Plugin]:
        """Gets all installed plugins. Do not modify the returned list."""
        return self.__plugins

