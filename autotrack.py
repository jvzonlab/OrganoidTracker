#!/usr/bin/env python3

"""Starts an empty window."""
from core import Experiment
from gui import launch_window, mainloop, plugin_loader
from visualizer.empty_visualizer import EmptyVisualizer
from visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
window.install_plugins(plugin_loader.load_plugins("plugins"))
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
