#!/usr/bin/env python3

"""Starts an empty window with the plugins from the Spindlesee folder.."""
from ai_track.core.experiment import Experiment
from ai_track.gui import plugin_loader
from ai_track.gui.launcher import launch_window, mainloop
from ai_track.visualizer.empty_visualizer import EmptyVisualizer
from ai_track.visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
window.install_plugins(plugin_loader.load_plugins("spindlesee_plugins"))
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
