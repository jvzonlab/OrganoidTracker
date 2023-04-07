import os.path
from typing import List, Iterable, Dict, Set

from organoid_tracker.core import UserError
from organoid_tracker.plugin import plugin_loader
from organoid_tracker.plugin.instance import Plugin
from organoid_tracker.plugin.registry import Registry

# Default location for user plugins
STANDARD_USER_PLUGIN_FOLDER = "~/OrganoidTracker/Plugins"
if os.name == 'nt':
    STANDARD_USER_PLUGIN_FOLDER = os.path.expandvars("%appdata%/OrganoidTracker/Plugins")


class PluginManager:
    """Holds all plugins in memory, as well as the folders where they were loaded from."""

    _folders: List[str]
    _built_in_folders: Set[str]  # Anything in here cannot be removed from self._folders

    _plugins: Dict[str, Plugin]  # Plugins, by absolute file
    _registry: Registry

    def __init__(self):
        self._plugins = dict()
        self._registry = Registry()
        self._folders = list()
        self._built_in_folders = set()

    def reload_plugins(self):
        """Reloads all plugins. Any new plugins in the folders will be picked up, any removed plugins will be
        unloaded."""
        self._registry.clear_all()

        # First, we try to reload any existing plugin
        old_plugins = self._plugins.copy()
        self._plugins.clear()
        for file_path, plugin in old_plugins.items():
            # We need to take care of any markers registered by the plugin
            # - first unload the old ones, then load the new ones
            if not os.path.exists(file_path):
                # Plugin was removed
                # We can't really unload it, but hopefully it'll be garbage collected
                continue
            directory = os.path.dirname(file_path)
            if directory not in self._folders:
                # Folder was removed - time to let the plugin go
                continue

            # Plugin still exists, reload it from disk
            plugin.reload()
            self._initialize_plugin(file_path, plugin)

        # Second, load any new plugins from the registered folders
        for folder in self._folders:
            self._load_new_plugins(folder)

    def _load_new_plugins(self, folder: str):
        """Loads any plugin in the given folder that wasn't loaded already."""
        if folder not in self._folders:
            raise ValueError("Can only load new plugins in registered folders")

        folder = os.path.abspath(folder)
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_path in self._plugins:
                continue  # Already loaded

            plugin = plugin_loader.load_plugin(file_path)
            if plugin is None:
                # Cannot be loaded
                continue

            self._initialize_plugin(file_path, plugin)

    def _initialize_plugin(self, file_path: str, plugin: Plugin):
        """Initializes the plugin, either for a new plugin or for a reloaded plugin."""
        self._plugins[file_path] = plugin
        for marker in plugin.get_markers():
            self._registry._markers[marker.save_name] = marker

    def get_plugins(self) -> Iterable[Plugin]:
        """Iterates over all loaded plugins."""
        yield from self._plugins.values()

    def load_folder(self, folder: str, *, built_in_folder: bool = False):
        """Loads a new folder. New plugins are loaded immediately.

        The folder argument is quite permissive. If it's empty or already in use, this method does nothing.
        Any environment variables in it, like "%appdata%", are expanded automatically.
        If the folder does not exist yet, it will be created.

        If built_in_folder is True, then the folder will not appear in get_user_folders().
        """
        if folder == "":
            return  # Empty, ignore.

        folder = os.path.abspath(os.path.expandvars(folder))
        if folder in self._folders:
            return  # Already in use

        os.makedirs(folder, exist_ok=True)
        self._folders.append(folder)
        if built_in_folder:
            self._built_in_folders.add(folder)
        self._load_new_plugins(folder)

    def get_user_folders(self) -> Iterable[str]:
        """Iterates over all plugin folders."""
        for folder in self._folders:
            if folder not in self._built_in_folders:
                yield folder

    def unregister_folder(self, folder: str):
        """Removes the folder from the list of folders. Will have no effect on loaded plugins until you reload the
        plugins."""
        if folder in self._folders:
            self._folders.remove(folder)

    def get_plugin_count(self) -> int:
        """Gets the number of loaded plugins."""
        return len(self._plugins)

    def get_registry(self) -> Registry:
        return self._registry

