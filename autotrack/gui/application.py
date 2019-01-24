from typing import Dict, Any, Set


class Plugin:
    """
    Represents a plugin. Plugins can add new data visualizers or provide support for more file types.
    Instead of writing a plugin, you can also write a script that uses the classes in core and imaging.
    """

    def init(self, window: "Window"):
        """Called once to run initialization code."""
        pass

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
