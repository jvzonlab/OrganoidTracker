"""Starts an empty window."""
from gui import launch_window, mainloop
from imaging import Experiment
from imaging.empty_visualizer import EmptyVisualizer
from imaging.visualizer import activate

experiment = Experiment()
window = launch_window(experiment)
visualizer = EmptyVisualizer(window)
activate(visualizer)
mainloop()
