#!/usr/bin/env python3

"""Starts an empty window."""
from core import Experiment
from gui import launch_window, mainloop
from visualizer.empty_visualizer import EmptyVisualizer
from visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
