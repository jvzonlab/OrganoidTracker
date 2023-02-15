import os.path
from typing import Dict, Any

from organoid_tracker.config import ConfigFile
from organoid_tracker.gui import action, dialog, APP_NAME
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return_dict = dict()

    for folder in window.plugin_manager.get_user_folders():
        return_dict["File//Plugins-Install new plugin//Folder-" + folder] = lambda: dialog.open_file(folder)

    return_dict["File//Plugins-Reload all plugins..."] = lambda: _reload_all_plugins(window)
    return_dict["File//Plugins-Install new plugin//Management-Add a new folder for plugins..."] = lambda: _add_folder(window)

    for folder in window.plugin_manager.get_user_folders():
        return_dict["File//Plugins-Install new plugin//Management-Remove a folder//Folder-" + folder] = lambda: _remove_folder(window, folder)

    return return_dict


def _reload_all_plugins(window: Window):
    window.plugin_manager.reload_plugins()
    window.redraw_all()
    window.set_status(f"Reloaded all {window.plugin_manager.get_plugin_count()} plugins.")


def _install_new_plugin():
    dialog.popup_message("Installing plugins",
                         "To install a plugin, open one of the plugin folders and add your plugin file there.")


def _add_folder(window: Window):
    if not dialog.popup_message_cancellable("New plugin folder", "With this option, you can select another folder"
                                                                 " that we will load plugins from. Afterwards, "
                                                                 + APP_NAME + " will load any \"plugin_\" files in"
                                                                 " that folder."):
        return
    folder = dialog.prompt_directory("New plugin folder")
    if folder is None:
        return
    window.plugin_manager.load_folder(folder)
    _store_folders(window)
    window.redraw_all()
    dialog.popup_message("Folder added",
                         f"The folder {folder} has been added as a plugin folder."
                         f"\n\nAny plugin placed into it is now active.")


def _store_folders(window: Window):
    """Stores the current plugin folders to the config file."""
    config = ConfigFile("scripts")
    config.set("extra_plugin_directory", os.path.pathsep.join(window.plugin_manager.get_user_folders()))
    config.save()


def _remove_folder(window: Window, folder: str):
    window.plugin_manager.unregister_folder(folder)
    window.plugin_manager.reload_plugins()
    _store_folders(window)
    window.redraw_all()
    dialog.popup_message("Folder removed",
                         f"The folder {folder} is no longer a plugin folder.\n\nAny plugin placed in it are no longer active.")
