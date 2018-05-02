import mahotas
import numpy
from matplotlib.backend_bases import KeyEvent

from gui import Window
from particle_detection import thresholding, watershedding
from segmentation import iso_intensity_curvature
from segmentation.iso_intensity_curvature import ImageDerivatives
from visualizer import activate, DisplaySettings
from visualizer.image_visualizer import AbstractImageVisualizer


class DetectionVisualizer(AbstractImageVisualizer):
    """Visualizer specialized in displaying particle positions.
    Use the left/right arrow keys to move in time.
    Use the up/down arrow keys to move in the z-direction
    Press D to perform 2D detection in this time point, showing intermediate results.
    Press N to show the next and current time point together in a single image (red=next time point, green=current)
    """

    color_map = "gray"

    def __init__(self, window: Window, time_point_number: int, z: int, display_settings: DisplaySettings):
        display_settings.show_next_time_point = False
        super().__init__(window, time_point_number, z, display_settings)

    def _draw_image(self):
        self._ax.imshow(self._time_point_images[self._z], cmap=self.color_map)

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "View": [
                "-",
                ("Show original images (T)", self.reset_view),
                ("Show basic threshold", self._basic_threshold),
                ("Add iso-intensity segmentation", self._segment_using_iso_intensity),
                ("Show advanced threshold (T)", self._advanced_threshold),
                "-",
                ("Perform cell detection", self._show_analysis),
                "-",
                ("Exit this view (/exit)", self._show_main_view)
            ]
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
        from visualizer.image_visualizer import StandardImageVisualizer
        v = StandardImageVisualizer(self._window, self._time_point.time_point_number(), self._z, self._display_settings)
        activate(v)

    def _basic_threshold(self):
        images = thresholding.image_to_8bit(self._experiment.get_image_stack(self._time_point))
        out = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.adaptive_threshold(images, out)

        self._time_point_to_rgb()
        self._time_point_images[:, :, :, 1] = out
        self._time_point_images[:, :, :, 2] = out
        self.draw_view()

    def _advanced_threshold(self):
        images = thresholding.image_to_8bit(self._experiment.get_image_stack(self._time_point))
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, threshold)

        self._time_point_to_rgb()
        self._time_point_images[:, :, :, 1] = threshold
        self._time_point_images[:, :, :, 2] = threshold
        self.draw_view()

    def _show_analysis(self):
        self.reset_view()

        images = thresholding.image_to_8bit(self._experiment.get_image_stack(self._time_point))
        threshold = numpy.empty_like(images, dtype=numpy.uint8)
        thresholding.advanced_threshold(images, threshold)
        self._time_point_images = threshold
        self.draw_view()

        distance_transform = numpy.empty_like(images, dtype=numpy.float64)
        watershedding.distance_transform(threshold, distance_transform)
        self._time_point_images = distance_transform
        self.draw_view()

        self._time_point_images = watershedding.watershed_maxima(threshold, distance_transform)
        self.color_map = watershedding.COLOR_MAP
        self.draw_view()

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

    def _segment_using_iso_intensity(self):
        images = self._experiment.get_image_stack(self._time_point)
        out = numpy.full_like(images, 255, dtype=numpy.uint8)
        iso_intensity_curvature.get_negative_gaussian_curvatures(images, ImageDerivatives(), out)

        self._time_point_to_rgb()
        self._time_point_images[:, :, :, 1] = self._time_point_images[:, :, :, 1] & out
        self._time_point_images[:, :, :, 2] = self._time_point_images[:, :, :, 2] & out
        self.draw_view()

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            if len(self._time_point_images.shape) == 4:
                # Reset view
                self.reset_view()
            else:
                # Show advanced threshold
                self._advanced_threshold()
            return
        super()._on_key_press(event)

    def reset_view(self):
        self.color_map = DetectionVisualizer.color_map
        self.refresh_view()