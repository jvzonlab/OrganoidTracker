from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from imaging import Experiment
from imaging.image_visualizer import AbstractImageVisualizer, activate
from particle_detection import distance_transform_detection
from typing import Dict

import matplotlib.pyplot as plt


def show(experiment: Experiment, detection_parameters: Dict):
    """Creates a visualizer suited particle positions for an experiment.
    Press S to view all detected positions at the current z"""
    figure = plt.figure(figsize=(8, 8))
    visualizer = DetectionVisualizer(experiment, figure, detection_parameters)
    activate(visualizer)


class DetectionVisualizer(AbstractImageVisualizer):
    """Visualizer specialized in displaying particle positions."""
    _detection_parameters = Dict

    def __init__(self, experiment: Experiment, figure: Figure, detection_parameters: Dict):
        super().__init__(experiment, figure)
        self._detection_parameters = detection_parameters

    def _on_key_press(self, event: KeyEvent):
        if event.key == "s":
            image = self._frame_images[self._z]
            distance_transform_detection.perform(image, show_results=True, **self._detection_parameters)
        super()._on_key_press(event)