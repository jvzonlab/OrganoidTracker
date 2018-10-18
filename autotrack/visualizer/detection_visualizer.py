from typing import Any, Tuple, Union, List

import cv2

import numpy
from numpy import ndarray
from matplotlib.backend_bases import KeyEvent

from autotrack.core import UserError
from autotrack.core.particles import Particle
from autotrack.gui import Window, dialog
from autotrack.particle_detection import thresholding, watershedding, gaussian_fit, smoothing, missed_cell_finder
from autotrack.core.gaussian import Gaussian
from autotrack.visualizer import activate, DisplaySettings
from autotrack.visualizer.image_visualizer import AbstractImageVisualizer


class DetectionVisualizer(AbstractImageVisualizer):
    """Visualizer specialized in displaying particle positions.
    """

    resolution = (2, 0.32, 0.32)
    minimal_size = (3, 11, 11)

    threshold_block_size = 51
    distance_transform_smooth_size = 21
    gaussian_fit_smooth_size = 11
    watershed_transform_smooth_size = 25

    def __init__(self, window: Window, time_point_number: int, z: int, display_settings: DisplaySettings):
        display_settings.show_next_time_point = False
        display_settings.show_reconstruction = False
        super().__init__(window, time_point_number, z, display_settings)

    def _get_window_title(self) -> str:
        return "Cell detection"

    def _must_show_other_time_points(self) -> bool:
        return False

    def _draw_error(self, particle: Particle, dz: int):
        pass  # Don't draw linking errors here, they are not interesting in this view

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "View/Show-Show original images (R)": self.refresh_view,
            "View/Exit-Exit this view (/exit)": self._show_main_view,
            "Threshold/Normal-Basic threshold": self._basic_threshold,
            "Threshold/Normal-With watershed segmentation": self.async(self._get_watershedded_threshold,
                                                                       self._display_threshold),
            "Threshold/Normal-With iso-intensity curvature segmentation": self.async(self._get_advanced_threshold,
                                                                                     self._display_threshold),
            "Threshold/Smoothed-Smoothed basic threshold": self.async(self._get_adaptive_smoothed_threshold,
                                                                      self._display_threshold),
            "Reconstruction/Default-Reconstruct normal treshold": self.async(self._get_threshold_reconstruction,
                                                                             self._display_watershed),
            "Reconstruction/Default-Reconstruct smoothed threshold": self.async(
                self._get_smoothed_threshold_reconstruction,
                self._display_watershed),
            "Reconstruction/Default-Reconstruct original image": self.async(
                self._get_image_reconstruction,
                self._display_reconstruction)
        }

    def _on_command(self, command: str):
        if command == "exit":
            self._show_main_view()
            return True
        if command == "help":
            self.update_status("Available commands:\n"
                               "/exit - Return to main view\.n"
                               "/t30 - Jump to time point 30. Also works for other time points.")

        return super()._on_command(command)

    def _show_main_view(self):
        from autotrack.visualizer.image_visualizer import StandardImageVisualizer
        v = StandardImageVisualizer(self._window, self._time_point.time_point_number(), self._z, self._display_settings)
        activate(v)

    def _basic_threshold(self):
        images = self._get_8bit_images()
        if images is None:
            dialog.popup_error("Failed to apply threshold", "Cannot show threshold - no images loaded.")
            return
        out = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.adaptive_threshold(images, out, self.threshold_block_size)

        self._display_threshold(out)

    def _get_watershedded_threshold(self) -> ndarray:
        images = self._get_8bit_images()
        if images is None:
            raise UserError("Failed to apply threshold", "Cannot show threshold - no images loaded.")
        images_smoothed = smoothing.get_smoothed(images, self.watershed_transform_smooth_size)
        out = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.watershedded_threshold(images, images_smoothed, out, self.threshold_block_size, self.minimal_size)

        return out

    def _display_threshold(self, out: ndarray):
        self._time_point_to_rgb()
        self._time_point_images[:, :, :, 1] = out
        self._time_point_images[:, :, :, 2] = out
        self.draw_view()

    def _get_advanced_threshold(self) -> ndarray:
        images = self._get_8bit_images()
        if images is None:
            raise UserError("Failed to apply threshold", "Cannot show threshold - no images loaded.")
        images_smoothed = smoothing.get_smoothed(images, self.watershed_transform_smooth_size)
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, images_smoothed, threshold, self.threshold_block_size, self.minimal_size)

        return threshold

    def _get_adaptive_smoothed_threshold(self) -> ndarray:
        images = self._get_8bit_images()
        if images is None:
            raise UserError("Failed to apply threshold", "Cannot show threshold - no images loaded.")
        images_smoothed = smoothing.get_smoothed(images, self.watershed_transform_smooth_size)
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.adaptive_threshold(images_smoothed, threshold, self.threshold_block_size)

        return threshold

    def _get_8bit_images(self):
        images = self._experiment.get_image_stack(self._time_point)
        if images is not None:
            return thresholding.image_to_8bit(images)
        return None

    def _display_image(self, image_stack: ndarray, color_map=None):
        self._time_point_images = image_stack
        self._color_map = color_map if color_map is not None else AbstractImageVisualizer._color_map
        self.draw_view()

    def _display_two_images(self, image_stacks: Tuple[ndarray, ndarray]):
        self._time_point_to_rgb()
        self._time_point_images[:, :, :, 0] = image_stacks[0]
        self._time_point_images[:, :, :, 1] = image_stacks[1]
        self._time_point_images[:, :, :, 2] = 0
        self.draw_view()

    def _display_watershed(self, image_stack: ndarray):
        self._display_image(image_stack, watershedding.COLOR_MAP)

    def _get_threshold_reconstruction(self, return_intermediate: bool = False) -> Any:
        particles = self._experiment.particles.of_time_point(self._time_point)
        if len(particles) == 0:
            raise UserError("Failed to detect cells", "Cannot detect cells - no particle positions loaded.")
        images = self._get_8bit_images()
        if images is None:
            raise UserError("Failed to detect cells", "Cannot detect cells - no images loaded.")

        images_smoothed = smoothing.get_smoothed(images, self.watershed_transform_smooth_size)
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, images_smoothed, threshold, self.threshold_block_size, self.minimal_size)

        # Labelling, calculate distance to label
        labels = numpy.empty_like(images, dtype=numpy.uint16)
        labels_count = len(particles)
        watershedding.create_labels(particles, labels)
        distance_transform_to_labels = watershedding.distance_transform_to_labels(labels, self.resolution)

        # Distance transform to edge and labels
        distance_transform = numpy.empty_like(images, dtype=numpy.float64)
        watershedding.distance_transform(threshold, distance_transform, self.resolution)
        smoothing.smooth(distance_transform, self.distance_transform_smooth_size)
        distance_transform += distance_transform_to_labels

        watershed = watershedding.watershed_labels(threshold, distance_transform.max() - distance_transform,
                                                   labels, labels_count)[0]
        self._print_missed_cells(watershed)

        if return_intermediate:
            return images, images_smoothed, watershed
        return watershed

    def _get_smoothed_threshold_reconstruction(self, return_intermediate=False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        images, images_smoothed, watershed = self._get_threshold_reconstruction(return_intermediate=True)

        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.adaptive_threshold(images_smoothed, threshold, self.threshold_block_size)

        ones = numpy.ones_like(images, dtype=numpy.uint8)
        watershed = watershedding.watershed_labels(threshold, ones, watershed, watershed.max())[0]
        if return_intermediate:
            return images, watershed
        return watershed

    def _get_image_reconstruction(self) -> List[Gaussian]:
        images, watershed = self._get_smoothed_threshold_reconstruction(return_intermediate=True)

        return gaussian_fit.perform_gaussian_mixture_fit_from_watershed(images, watershed, self.gaussian_fit_smooth_size)

    def _display_reconstruction(self, gaussians: List[Gaussian]):
        self._experiment.remove_particles(self._time_point)
        shape = self._time_point_images.shape  # May be 3D or 4D, depending on what was previously displayed
        canvas = numpy.zeros((shape[0], shape[1], shape[2], 3), dtype=numpy.float64)

        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        i = 0
        for gaussian in gaussians:
            if gaussian is None:
                continue
            self._experiment.add_particle(
                Particle(gaussian.mu_x, gaussian.mu_y, gaussian.mu_z).with_time_point(self._time_point))
            color = colors[i % len(colors)]
            gaussian.draw_colored(canvas, color)
            i += 1

        canvas.clip(0, 1, out=canvas)
        self._display_image(canvas)

    def _time_point_to_rgb(self):
        """If the time point image is a black-and-white image, it is converted to RGB"""
        shape = self._time_point_images.shape
        if len(shape) == 4:
            # Already a full-color image
            return

        old = self._time_point_images
        self._time_point_images = numpy.zeros((shape[0], shape[1], shape[2], 3), dtype=numpy.uint8)
        self._time_point_images[:, :, :, 0] = old / old.max() * 255
        self._time_point_images[:, :, :, 1] = self._time_point_images[:, :, :, 0]
        self._time_point_images[:, :, :, 2] = self._time_point_images[:, :, :, 0]

    def _on_key_press(self, event: KeyEvent):
        if event.key == "r":
            # Reset view
            self.refresh_view()
        super()._on_key_press(event)

    def _print_missed_cells(self, watershed: ndarray):
        particles = self._experiment.particles.of_time_point(self._time_point)
        if len(particles) == 0:
            return
        errors = missed_cell_finder.find_undetected_particles(watershed, particles)
        for particle, error in errors.items():
            print("Error at " + str(particle) + ": " + str(error))
