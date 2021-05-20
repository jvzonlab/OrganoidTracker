#!/usr/bin/env python3

"""Starts an empty window with the plugins from the OrganoidTracker folder."""
import os.path
import sys

from organoid_tracker.config import ConfigFile
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.plugin import plugin_loader
from organoid_tracker.gui.launcher import launch_window, mainloop
from organoid_tracker.visualizer.empty_visualizer import EmptyVisualizer
from organoid_tracker.visualizer import activate

# Load plugins
directory = os.path.dirname(os.path.abspath(__file__))
plugin_directory = os.path.join(directory, "organoid_tracker_plugins")
plugins = plugin_loader.load_plugins(plugin_directory)

# Load extra plugins (we don't save the config, otherwise you would end up with a configuration file in every directory
# where you run the visualizer)
config = ConfigFile("visualizer")
extra_plugin_directory = config.get_or_default("extra_plugin_directory", "")
if extra_plugin_directory != "":
    plugins += plugin_loader.load_plugins(extra_plugin_directory)

# Command handling
if len(sys.argv) > 1:
    command = sys.argv[1]
    for plugin in plugins:
        registered_commands = plugin.get_commands()
        if command in registered_commands:
            exit(registered_commands[command](sys.argv[2:]))

# Open window
experiment = Experiment()
window = launch_window(experiment)
window.install_plugins(plugins)

visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
