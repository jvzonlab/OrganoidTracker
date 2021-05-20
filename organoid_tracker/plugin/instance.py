"""Module containing plugin-related code."""
from typing import Dict, Any, Set, TYPE_CHECKING, List, Callable

from organoid_tracker.core.marker import Marker

if TYPE_CHECKING:
    from organoid_tracker.gui.window import Window


class Plugin:
    """
    Represents a plugin. Plugins can add new data visualizers or provide support for more file types.
    Instead of writing a plugin, you can also write a script that uses the classes in core and imaging.
    """

    def get_markers(self) -> List[Marker]:
        """Called once to run initialization code."""
        pass

    def get_menu_items(self, window: "Window") -> Dict[str, Callable[[], None]]:
        """
        Used to add menu items that must always be visible. Example:

            return {
                "File//Import-Import my format... [Ctrl+W]": lambda: my_code_here(),
                "View//Analysis-Useful analysis screen here...": lambda: my_other_code_here()
            }
        """
        return {}

    def get_commands(self) -> Dict[str, Callable[[List[str]], int]]:
        """Used to add new command-line tools. Returns a dict: {"COMMAND NAME": callable(args)}.
        Those are called using `python organoid_tracker.py [COMMAND NAME] [ARGS]`.
        Command names are case sensitive and should be lower case.
        The command callable takes a list of args, and returns a status code: 0 for success,
        anything else is an error code. This error code is passed to the operating system."""
        return {}

    def reload(self):
        """Reloads this plugin from disk."""
        ...
