#!/usr/bin/env python3

"""Starts an empty window with the plugins from the OrganoidTracker folder."""
import os.path
import sys

from organoid_tracker.config import ConfigFile
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.plugin import plugin_loader, plugin_manager
from organoid_tracker.gui.launcher import launch_window, mainloop
from organoid_tracker.plugin.plugin_manager import PluginManager
from organoid_tracker.visualizer.empty_visualizer import EmptyVisualizer
from organoid_tracker.visualizer import activate

# Load plugins
directory = os.path.dirname(os.path.abspath(__file__))
plugin_directory = os.path.join(directory, "organoid_tracker_plugins")
plugins = PluginManager()
plugins.load_folder(plugin_directory, built_in_folder=True)
# Load extra plugins (we don't save the config, otherwise you would end up with a configuration file in every directory
# where you run the visualizer)
config = ConfigFile("scripts")
for extra_plugin_directory in config.get_or_default("extra_plugin_directory", plugin_manager.STANDARD_USER_PLUGIN_FOLDER).split(os.path.pathsep):
    plugins.load_folder(extra_plugin_directory)

# Command handling
if len(sys.argv) > 1:
    command = sys.argv[1]
    for plugin in plugins.get_plugins():
        registered_commands = plugin.get_commands()
        if command in registered_commands:
            exit(registered_commands[command](sys.argv[2:]))
    raise ValueError("Invalid command: " + command)

# Open window
experiment = Experiment()
window = launch_window(experiment)
window.replace_plugin_manager(plugins)

visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
