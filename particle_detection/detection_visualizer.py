from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from imaging import Experiment
from imaging.image_visualizer import AbstractImageVisualizer, activate
from particle_detection import Detector


def show(experiment: Experiment, detector: Detector, detection_parameters: Dict):
    """Creates a visualizer suited particle positions for an experiment.
    Press S to view all detected positions at the current z"""
    figure = plt.figure(figsize=(8, 8))
    visualizer = DetectionVisualizer(experiment, figure, detector, detection_parameters)
    activate(visualizer)


class DetectionVisualizer(AbstractImageVisualizer):
    """Visualizer specialized in displaying particle positions.
    Press D to perform 2D detection in this frame, showing intermediate results."""
    _detection_parameters = Dict
    _detector: Detector

    def __init__(self, experiment: Experiment, figure: Figure, detector: Detector, detection_parameters: Dict):
        super().__init__(experiment, figure)
        self._detection_parameters = detection_parameters
        self._detector = detector

    def _on_key_press(self, event: KeyEvent):
        image = self._frame_images[self._z]
        if event.key == "d":
            self._detector.detect(image, show_results=True, **self._detection_parameters)

        super()._on_key_press(event)