#!/usr/bin/env python3

"""Starts an empty window with the plugins from the Spindlesee folder.."""
from autotrack.core.experiment import Experiment
from autotrack.gui import plugin_loader
from autotrack.gui.launcher import launch_window, mainloop
from autotrack.visualizer.empty_visualizer import EmptyVisualizer
from autotrack.visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
window.install_plugins(plugin_loader.load_plugins("spindlesee_plugins"))
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
