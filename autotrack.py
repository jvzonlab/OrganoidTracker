#!/usr/bin/env python3

"""Starts an empty window."""
from autotrack.core.experiment import Experiment
from autotrack.gui import plugin_loader
from autotrack.gui.launcher import launch_window, mainloop
from autotrack.visualizer.empty_visualizer import EmptyVisualizer
from autotrack.visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
window.install_plugins(plugin_loader.load_plugins("autotrack_plugins"))
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
