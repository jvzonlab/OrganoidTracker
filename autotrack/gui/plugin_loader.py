import os
from typing import Any, List, Dict
import importlib

from autotrack.gui import Plugin, Window


class _FilePlugin(Plugin):
    """A plugin that consists of a single .py file."""
    __loaded_script: Any
    __module_name: str

    def __init__(self, module_name: str):
        self.__module_name = module_name
        self.__loaded_script = importlib.import_module(module_name)

    def get_menu_items(self, window: Window):
        if hasattr(self.__loaded_script, 'get_menu_items'):
            return self.__loaded_script.get_menu_items(window)
        return {}

    def reload(self):
        importlib.reload(self.__loaded_script)


def load_plugins(folder: str) -> List[Plugin]:
    """Loads the plugins in the given folder. The folder must follow the format "example/folder/structure". A plugin in
    "example/folder/structure/my_plugin.py" will then be loaded as the module "example.folder.structure.my_plugin".
    """
    if not os.path.exists(folder):
        print("No plugins folder found at " + os.path.abspath(folder))
        return []

    module_prefix = folder.replace("/", ".")
    if not module_prefix.endswith("."):
        module_prefix += "."

    plugins = []
    for dir_entry in os.scandir(folder):
        file_name = dir_entry.name
        if not file_name.startswith("plugin_"):
            if file_name.endswith(".py"):
                print("Ignoring file " + file_name + " in " + folder + " folder: it does not start with \"plugin_\"")
            continue
        if dir_entry.is_dir():
            print("Ignoring file " + file_name + " in " + folder + " folder: it is a directory")
            continue

        module_name = module_prefix + file_name.replace(".py", "")
        plugins.append(_FilePlugin(module_name))
    return plugins
