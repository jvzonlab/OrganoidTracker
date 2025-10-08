import os
import sys
from typing import Any, List, Dict, Tuple, Callable, Optional
import importlib

from organoid_tracker.core.marker import Marker
from organoid_tracker.imaging.file_loader import FileLoader
from organoid_tracker.plugin.instance import Plugin


class _ModulePlugin(Plugin):
    """A plugin that consists of a single .py file."""
    _loaded_module_name: str
    _loaded_script: Any

    def __init__(self, file_name: str):
        self._loaded_module_name = _to_module_name(file_name)
        self._loaded_script = importlib.import_module(self._loaded_module_name)

    def get_markers(self) -> List[Marker]:
        if hasattr(self._loaded_script, 'get_markers'):
            return self._loaded_script.get_markers()
        return []

    def get_menu_items(self, window: "Window") -> Dict[str, Callable[[], None]]:
        if hasattr(self._loaded_script, 'get_menu_items'):
            return self._loaded_script.get_menu_items(window)
        return {}

    def get_file_loaders(self) -> List[FileLoader]:
        if hasattr(self._loaded_script, 'get_file_loaders'):
            return self._loaded_script.get_file_loaders()
        return []

    def reload(self):
        importlib.reload(self._loaded_script)

        to_unload_prefix = self._loaded_module_name + "."
        for module_name in list(sys.modules.keys()):
            # Reload submodules
            if module_name.startswith(to_unload_prefix):
                importlib.reload(sys.modules[module_name])

    def get_commands(self) -> Dict[str, Callable[[str], int]]:
        if hasattr(self._loaded_script, 'get_commands'):
            return self._loaded_script.get_commands()
        return {}


def _to_module_name(file: str) -> str:
    """Returns the module name for the given file. A file stored in example_folder/test.py will end up as the module
    `example_folder.test`. In this way, relative imports still work fine. Returns the module name."""
    file = os.path.abspath(file)
    if not file.endswith(".py") and not os.path.exists(os.path.join(file, "__init__.py")):
        raise ValueError("Not a Python file or module: " + file)
    parent_folder = os.path.dirname(file)
    grandparent_folder = os.path.dirname(parent_folder)

    # Add to path
    if grandparent_folder not in sys.path:
        sys.path.insert(0, grandparent_folder)

    # Load module
    file_name = os.path.basename(file)
    module_name = os.path.basename(parent_folder) + "." + \
                  (file_name[:-len(".py")] if file_name.endswith(".py") else file_name)
    return module_name


def load_plugin(file_path: str) -> Optional[Plugin]:
    """Loads a single plugin. The file_path can point to a single Python file or to a folder that is a Python module.
    The file name must be the full path, the basename must start with 'plugin_'.

    Returns None if there is no plugin at that location."""
    file_name = os.path.basename(file_path)
    if not file_name.startswith("plugin_"):
        if file_name.endswith(".py"):
            print("Ignoring Python file " + file_name + " in " + os.path.basename(file_path)
                  + " folder: it does not start with \"plugin_\"")
        return None
    if os.path.isdir(file_path):
        return _ModulePlugin(file_path)
    elif file_name.endswith(".py"):
        return _ModulePlugin(file_path)
    else:
        print("Ignoring file " + file_name + " in " + os.path.dirname(file_path)
              + " folder: is looks like a plugin, but is not a folder or a Python file.")
        return None
