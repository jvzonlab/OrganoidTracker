from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from imaging import Experiment
from imaging.image_visualizer import AbstractImageVisualizer, activate
from particle_detection import dt_detection, edge_detection, dt_with_laplacian
from typing import Dict

import matplotlib.pyplot as plt

from particle_detection.dt_detection import DistanceTransformDetector
from particle_detection.dt_with_laplacian import DistanceTransformWithLaplacianDetector


def show(experiment: Experiment, detection_parameters: Dict):
    """Creates a visualizer suited particle positions for an experiment.
    Press S to view all detected positions at the current z"""
    figure = plt.figure(figsize=(8, 8))
    visualizer = DetectionVisualizer(experiment, figure, detection_parameters)
    activate(visualizer)


class DetectionVisualizer(AbstractImageVisualizer):
    """Visualizer specialized in displaying particle positions.
    Press M to perform local minima detection
    Press E to perform edge detection"""
    _detection_parameters = Dict

    def __init__(self, experiment: Experiment, figure: Figure, detection_parameters: Dict):
        super().__init__(experiment, figure)
        self._detection_parameters = detection_parameters

    def _on_key_press(self, event: KeyEvent):
        image = self._frame_images[self._z]
        if event.key == "d":
            DistanceTransformDetector().detect(image, show_results=True, **self._detection_parameters)
        if event.key == "e":
            edge_detection.perform(image)
        if event.key == "l":
            DistanceTransformWithLaplacianDetector().detect(image, show_results=True, **self._detection_parameters)

        super()._on_key_press(event)