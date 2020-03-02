#!/usr/bin/env python3

"""Starts an empty window with the plugins from the OrganoidTracker folder."""
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import plugin_loader
from organoid_tracker.gui.launcher import launch_window, mainloop
from organoid_tracker.visualizer.empty_visualizer import EmptyVisualizer
from organoid_tracker.visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
window.install_plugins(plugin_loader.load_plugins("organoid_tracker_plugins"))
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
