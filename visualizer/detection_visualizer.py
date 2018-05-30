from time import sleep
from typing import Any, Tuple, Optional

import cv2

import numpy
from numpy import ndarray
from matplotlib.backend_bases import KeyEvent

from core import UserError, Particle
from gui import Window, dialog
from particle_detection import thresholding, watershedding, missed_cell_finder, smoothing, gaussian_fit
from visualizer import activate, DisplaySettings
from visualizer.image_visualizer import AbstractImageVisualizer


class DetectionVisualizer(AbstractImageVisualizer):
    """Visualizer specialized in displaying particle positions.
    """

    threshold_block_size = 51
    sampling = (2, 0.32, 0.32)
    minimal_size = (3, 11, 11)
    distance_transform_smooth_size = 21

    color_map = "gray"

    def __init__(self, window: Window, time_point_number: int, z: int, display_settings: DisplaySettings):
        display_settings.show_next_time_point = False
        display_settings.show_shapes = False
        super().__init__(window, time_point_number, z, display_settings)

    def _draw_image(self):
        if self._time_point_images is not None:
            self._ax.imshow(self._time_point_images[self._z], cmap=self.color_map)

    def _get_window_title(self) -> str:
        return "Cell detection"

    def _must_show_other_time_points(self) -> bool:
        return False

    def _draw_error(self, particle: Particle, dz: int):
        pass  # Don't draw linking errors here, they are not interesting in this view

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "View": [
                ("Show original images (R)", self.refresh_view),
                "-",
                ("Exit this view (/exit)", self._show_main_view)
            ],
            "Threshold": [
                ("Basic threshold", self._basic_threshold),
                ("With watershed segmentation", self.async(self._get_watershedded_threshold, self._display_threshold)),
                ("With iso-intensity curvature segmentation", self.async(self._get_advanced_threshold,
                                                                             self._display_threshold)),
            ],
            "Detection": [
                ("Detect cells", self.async(self._get_detected_cells, self._display_watershed)),
                ("Detect cells using existing points", self.async(self._get_detected_cells_using_particles,
                                                                  self._display_watershed)),
                ("Detect contours", self.async(self._get_detected_contours, self._display_threshold)),
            ],
            "Reconstruction": [
                ("Reconstruct basic threshold using existing points", self.async(self._get_reconstruction_of_basic_threshold,
                                                                       self._display_watershed)),
                ("Reconstruct cells using existing points", self.async(self._get_reconstruction_using_particles,
                                                                       self._display_watershed))
            ]
        }

    def _on_command(self, command: str):
        if command == "exit":
            self._show_main_view()
            return True
        if command == "help":
            self._update_status("Available commands:\n"
                               "/exit - Return to main view\.n"
                               "/t30 - Jump to time point 30. Also works for other time points.")

        return super()._on_command(command)

    def _show_main_view(self):
        from visualizer.image_visualizer import StandardImageVisualizer
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
        images_smoothed = smoothing.get_smoothed(images, int(self.threshold_block_size / 2))
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
        images_smoothed = smoothing.get_smoothed(images, int(self.threshold_block_size / 2))
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, images_smoothed, threshold, self.threshold_block_size, self.minimal_size)

        return threshold

    def _get_8bit_images(self):
        images = self._experiment.get_image_stack(self._time_point)
        if images is not None:
            return thresholding.image_to_8bit(images)
        return None

    def _display_image(self, image_stack: ndarray, color_map=None):
        self._time_point_images = image_stack
        self.color_map = color_map if color_map is not None else DetectionVisualizer.color_map
        self.draw_view()

    def _display_two_images(self, image_stacks: Tuple[ndarray, ndarray]):
        self._time_point_to_rgb()
        self._time_point_images[:, :, :, 0] = image_stacks[0]
        self._time_point_images[:, :, :, 1] = image_stacks[1]
        self._time_point_images[:, :, :, 2] = 0
        self.draw_view()

    def _display_watershed(self, image_stack: ndarray):
        self._display_image(image_stack, watershedding.COLOR_MAP)

    def _get_detected_cells(self, return_intermediate: bool = False):
        images = self._get_8bit_images()
        if images is None:
            dialog.popup_error("Failed to detect cells", "Cannot detect cells - no images loaded.")
            return

        images_smoothed = smoothing.get_smoothed(images, int(self.threshold_block_size / 2))
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, images_smoothed, threshold, self.threshold_block_size, self.minimal_size)

        distance_transform = numpy.empty_like(images, dtype=numpy.float64)
        watershedding.distance_transform(threshold, distance_transform, self.sampling)

        watershed = watershedding.watershed_maxima(threshold, distance_transform, self.minimal_size)[0]
        self._print_missed_cells(watershed)
        if return_intermediate:
            return images, images_smoothed, watershed
        return watershed

    def _get_detected_cells_using_particles(self, return_intermediate: bool = False) -> Any:
        if len(self._time_point.particles()) == 0:
            dialog.popup_error("Failed to detect cells", "Cannot detect cells - no particle positions loaded.")
            return
        images = self._get_8bit_images()
        if images is None:
            dialog.popup_error("Failed to detect cells", "Cannot detect cells - no images loaded.")
            return

        images_smoothed = smoothing.get_smoothed(images, int(self.threshold_block_size / 2))
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, images_smoothed, threshold, self.threshold_block_size, self.minimal_size)

        # Labelling, calculate distance to label
        particles = self._time_point.particles()
        labels = numpy.empty_like(images, dtype=numpy.uint16)
        labels_count = len(particles)
        watershedding.create_labels(particles, labels)
        distance_transform_to_labels = self._get_distances_to_labels(images, labels)

        # Distance transform to edge and labels
        distance_transform = numpy.empty_like(images, dtype=numpy.float64)
        watershedding.distance_transform(threshold, distance_transform, self.sampling)
        smoothing.smooth(distance_transform, self.distance_transform_smooth_size)
        distance_transform += distance_transform_to_labels

        watershed = watershedding.watershed_labels(threshold, distance_transform.max() - distance_transform,
                                                   labels, labels_count)[0]
        self._print_missed_cells(watershed)

        if return_intermediate:
            return images, images_smoothed, watershed
        return watershed

    def _get_reconstruction_of_basic_threshold(self) -> ndarray:
        images, images_smoothed, watershed = self._get_detected_cells_using_particles(return_intermediate=True)

        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.adaptive_threshold(images_smoothed, threshold, self.threshold_block_size)

        ones = numpy.ones_like(images, dtype=numpy.uint8)
        return watershedding.watershed_labels(threshold, ones, watershed, watershed.max())[0]

    def _get_reconstruction_using_particles(self) -> ndarray:
        watershed = self._get_reconstruction_of_basic_threshold()

        self._time_point_to_rgb()
        result = self._time_point_images
        gaussian_fit.perform_gaussian_mixture_fit_from_watershed(watershed, result)
        return result

    def _get_distances_to_labels(self, images, labels):
        labels_inv = numpy.full_like(images, 255, dtype=numpy.uint8)
        labels_inv[labels != 0] = 0
        distance_transform_to_labels = numpy.empty_like(images, dtype=numpy.float64)
        watershedding.distance_transform(labels_inv, distance_transform_to_labels, self.sampling)
        distance_transform_to_labels[distance_transform_to_labels > 4] = 4
        distance_transform_to_labels = 4 - distance_transform_to_labels
        return distance_transform_to_labels

    def _get_detected_contours(self):
        images = self._get_8bit_images()
        if images is None:
            dialog.popup_error("Failed to detect cells", "Cannot detect cells - no images loaded.")
            return

        images_smoothed = smoothing.get_smoothed(images, int(self.threshold_block_size / 2))
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, images_smoothed, threshold, self.threshold_block_size, self.minimal_size)

        im2, contours, hierarchy = cv2.findContours(threshold[self._z], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        threshold[self._z] = 0
        for contour in contours:
            cv2.drawContours(threshold[self._z], [contour], 0, 255, 2)
        return threshold

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

    def refresh_view(self):
        self.color_map = DetectionVisualizer.color_map
        super().refresh_view()

    def _print_missed_cells(self, watershed: ndarray):
        particles = self._time_point.particles()
        if len(particles) == 0:
            return
        errors = missed_cell_finder.find_undetected_particles(watershed, particles)
        for particle, error in errors.items():
            print("Error at " + str(particle) + ": " + str(error))