import os
from typing import Dict, Any, Iterable, NamedTuple

import numpy
import skimage.draw
from matplotlib.colors import Colormap

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob


class _MovieSettings(NamedTuple):
    """Settings for the movie generation. For now, they are hardcoded."""
    xy_upscaling_factor: int = 4

    tracking_line_width: int = 1
    tracking_tail_length: int = 5



def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export movie//Projection-Max-intensity projection...":
            lambda: _generate_max_projection_movie(window),
    }


def _generate_max_projection_movie(window: Window):
    experiments_open = len(window.get_gui_experiment().get_active_tabs())
    if experiments_open == 0:
        raise UserError("No experiments open", "Please open an experiment before generating a movie.")

    if experiments_open == 1:
        # Save to a single file
        output_file_or_folder = dialog.prompt_save_file("Save movie as...", [("TIFF files", "*.tif")])
        if output_file_or_folder is None:
            return
    else:
        # Save to a folder
        output_file_or_folder = dialog.prompt_save_file("Select folder to save movies in...", [("Folders", "*")])
        if output_file_or_folder is None:
            return
        os.makedirs(output_file_or_folder, exist_ok=True)

    job = _MovieJob(output_file_or_folder=output_file_or_folder)
    worker_job.submit_job(window, job)
    window.set_status(f"Creating max projection movie for {experiments_open} experiment(s)...")


class _MovieJob(WorkerJob):

    output_file_or_folder: str
    settings: _MovieSettings

    def __init__(self, output_file_or_folder: str):
        self.output_file_or_folder = output_file_or_folder
        self.settings = _MovieSettings()

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, positions=True, links=True, name=True)

    def gather_data(self, experiment_copy: Experiment) -> Any:
        # Build array for storing the movie frames
        image_size_zyx = experiment_copy.images.image_loader().get_image_size_zyx()
        if image_size_zyx is None:
            raise UserError("No image size", "The image size could not be determined. Please check that the images are loaded correctly, and are of the same size.")
        last_time_point_number = experiment_copy.last_time_point_number()
        first_time_point_number = experiment_copy.first_time_point_number()
        if last_time_point_number is None or first_time_point_number is None:
            raise UserError("No time points", "The experiment has no time points. Please check that the images are loaded correctly.")
        time_point_count = last_time_point_number - first_time_point_number + 1
        canvas_rgb = numpy.zeros((time_point_count, image_size_zyx[1] * self.settings.xy_upscaling_factor,
                                  image_size_zyx[2] * self.settings.xy_upscaling_factor, 3), dtype=numpy.uint8)

        # Create the movie
        canvas_rgb_t = numpy.zeros((image_size_zyx[1] * self.settings.xy_upscaling_factor, image_size_zyx[2] * self.settings.xy_upscaling_factor, 3), dtype=numpy.uint16)  # Temporary canvas for each time point
        tracking_canvas_rgb_t = numpy.zeros_like(canvas_rgb_t, dtype=numpy.uint16)  # Temporary canvas for tracking data
        for time_point in self.reporting_progress(experiment_copy.time_points()):
            self._draw_images_of_time_point(experiment_copy, time_point, canvas_rgb_t)
            self._draw_tracking_data_of_time_point(experiment_copy, time_point, tracking_canvas_rgb_t)

            # Merge both canvases together
            canvas_rgb_t += tracking_canvas_rgb_t
            numpy.clip(canvas_rgb_t, 0, 255, out=canvas_rgb_t)  # Ensure values are in [0, 255]

            canvas_rgb[time_point.time_point_number() - first_time_point_number] = canvas_rgb_t

        return canvas_rgb

    def use_data(self, tab: SingleGuiTab, data: Any):
        # Write out the data
        import tifffile
        if os.path.isdir(self.output_file_or_folder):
            experiment_name = tab.experiment.name.get_save_name()
            output_file = os.path.join(self.output_file_or_folder, f"{experiment_name}.tif")
            tifffile.imwrite(output_file, data, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})
        else:
            tifffile.imwrite(self.output_file_or_folder, data, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

    def on_finished(self, data: Iterable[Any]):
        dialog.popup_message("Movie created", f"Done! The movie has been saved to {self.output_file_or_folder}.")


    def _draw_images_of_time_point(self, experiment: Experiment, time_point: TimePoint, canvas_rgb: numpy.ndarray) -> numpy.ndarray:
        """Draws the max projection of all channels for a given time point onto the provided canvas_rgb array.

        The size of the array must match the size of the max projection images, and be of type uint16 to avoid overflow
        when summing the channels. However, only the range [0, 255] will be used for the final output, so values will be
        clipped to this range.
        """

        canvas_rgb[...] = 0  # Reset canvas to black

        for image_channel in experiment.images.get_channels():
            colormap: Colormap = experiment.images.get_channel_description(image_channel).colormap

            image_3d = experiment.images.get_image_stack(time_point, image_channel)
            max_projection = numpy.max(image_3d, axis=0)

            # Normalize to 0-1
            max_projection_normalized = (max_projection - max_projection.min()) / (max(numpy.ptp(max_projection), 1e-8))

            # Apply colormap
            max_projection_normalized = colormap(max_projection_normalized)[:, :, :3]  # Drop alpha channel

            # Convert to 0-255
            max_projection_normalized = (max_projection_normalized * 255).astype(numpy.uint8)

            # Scale to the canvas size and add
            canvas_rgb += skimage.transform.resize(max_projection_normalized, (canvas_rgb.shape[0], canvas_rgb.shape[1], 3), order=0)
            canvas_rgb.clip(0, 255, out=canvas_rgb)  # Ensure values are in [0, 255]


    def _draw_tracking_data_of_time_point(self, experiment: Experiment, time_point: TimePoint, canvas_rgb: numpy.ndarray) -> None:
        """Draws the tracking data for a given time point onto the provided canvas_rgb array."""
        canvas_rgb[...] = 0  # Reset canvas to black

        tracking_brightnesses = numpy.linspace(0, 1, self.settings.tracking_tail_length + 1, endpoint=True)[1:]
        tracking_brightnesses **= 2  # To have the brightness decrease faster for older tracking data
        tracking_brightnesses *= 255  # Scale from 0 to 255

        for i in range(self.settings.tracking_tail_length):
            # Draw oldest first, so that the most recent tracking data is on top
            time_points_back = len(tracking_brightnesses) - 1 - i
            past_time_point = time_point - time_points_back
            brightness = tracking_brightnesses[i]
            color = (brightness, brightness, brightness)
            self._draw_tracking_data_one_time_point(experiment, past_time_point, canvas_rgb, color)


    def _draw_tracking_data_one_time_point(self, experiment: Experiment, time_point: TimePoint, canvas_rgb: numpy.ndarray, color: tuple[int, int, int]) -> None:
        """Draws the tracking data for a given time point onto the provided canvas_rgb array.
        """

        offset_current = experiment.images.offsets.of_time_point(time_point)
        offset_previous = experiment.images.offsets.of_time_point(time_point - 1)

        line_width = self.settings.tracking_line_width

        canvas_lines_bool = numpy.zeros((canvas_rgb.shape[0], canvas_rgb.shape[1]), dtype=bool)
        for position_current in experiment.positions.of_time_point(time_point):
            x_current = round((position_current.x - offset_current.x) * self.settings.xy_upscaling_factor)
            y_current = round((position_current.y - offset_current.y) * self.settings.xy_upscaling_factor)

            for position_previous in experiment.links.find_pasts(position_current):
                x_prev = round((position_previous.x - offset_previous.x) * self.settings.xy_upscaling_factor)
                y_prev = round((position_previous.y - offset_previous.y) * self.settings.xy_upscaling_factor)

                # Draw a line between the previous and current position
                rr, cc = skimage.draw.line(y_prev, x_prev, y_current, x_current)
                canvas_lines_bool[rr, cc] = True

        # Make the lines thicker by dilating the boolean mask
        circle_element = skimage.morphology.disk(line_width)
        canvas_lines_bool = skimage.morphology.dilation(canvas_lines_bool, footprint=circle_element)

        # Set the pixels of the lines to white
        canvas_rgb[canvas_lines_bool] = color
