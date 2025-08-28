from typing import Dict, Any, Iterable, Optional

import numpy
import tifffile
from matplotlib.backend_bases import MouseEvent

from organoid_tracker import core
from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.imaging import cropper
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window):
    return {
        "File//Export-Export movie//Cell-Single-cell movie...": lambda: _export_single_cell_movie(window)
    }


def _export_single_cell_movie(window: Window):
    activate(_SingleCellMovieVisualizer(window))


class _SingleCellMovieVisualizer(ExitableImageVisualizer):
    """Click on a position to select it. Then use the Movie menu to set the parameters and export a movie of this cell."""

    _crop_size_xy_px: int = 80
    _selected_position: Optional[Position] = None

    def __init__(self, window: Window):
        super().__init__(window)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Movie//Parameters-Set crop size...": self._set_crop_size,
            "Movie//Export-Export movie...": self._export_movie
        }

    def _get_figure_title(self) -> str:
        channel_name = self._experiment.images.get_channel_description(
            self._display_settings.image_channel).channel_name

        return (f"Single-cell movie exporter\n"
                f"Time point {self._time_point.time_point_number()}    (z={self._get_figure_title_z_str()}, c={channel_name})")


    def _draw_extra(self):
        # Draw the selected position
        if self._selected_position is None:
            return

        dt = self._selected_position.time_point_number() - self._time_point.time_point_number()
        if abs(dt) > 3:
            return  # Only draw if within 3 time points
        color = core.COLOR_CELL_CURRENT
        if dt < 0:
            color = core.COLOR_CELL_PREVIOUS
        elif dt > 0:
            color = core.COLOR_CELL_NEXT
        self._draw_selection(self._selected_position, color)

    def _on_mouse_single_click(self, event: MouseEvent):
        if event.button != 1:
            return  # Only left click

        clicked_position = self._get_position_at(event.xdata, event.ydata)
        if clicked_position is None:
            self._selected_position = None
            self.draw_view()
            self.update_status("No position at this location. Unselected any previously selected position.")
            return

        if self._experiment.links.get_track(clicked_position) is None:
            self._selected_position = None
            self.draw_view()
            self.update_status("The clicked position is not part of any track, so we cannot make a movie for this position.")
            return

        self._selected_position = clicked_position
        self.draw_view()
        self.update_status(f"Selected {self._selected_position}. Use the Movie menu to export a movie of this cell.")


    def _set_crop_size(self):
        crop_size_xy = dialog.prompt_int("Crop size", "What should be the height and width of the crops in pixels?",
                                         minimum=5, default=self._crop_size_xy_px, maximum=2000)
        if crop_size_xy is not None:
            self._crop_size_xy_px = crop_size_xy

    def _export_movie(self):
        if self._selected_position is None:
            raise UserError("No position selected", "Please select a position first by clicking on it.")

        output_file = dialog.prompt_save_file("Movie location", [("TIFF file", "*.tif")])
        if output_file is None:
            return  # Cancelled

        if not self._experiment.images.image_loader().has_images():
            raise UserError("No images available", "The experiment has no images. Cannot create a movie.")

        track = self._experiment.links.get_track(self._selected_position)
        if track is None:
            raise UserError("Position not tracked", "The selected position is not part of any track. Cannot create a movie.")

        worker_job.submit_job(self._window, _ExportMovieJob(
            position=self._selected_position,
            crop_size_xy_px=self._crop_size_xy_px,
            channel=self._display_settings.image_channel,
            output_file=output_file
        ))
        self.update_status("Started creating the movie...")


def _find_average_position(origin_track: LinkingTrack, time_point: TimePoint) -> Optional[Position]:
    """Finds the average position of all offspring/predecessor cells related to origin_track at the given time point."""
    if origin_track.first_time_point() <= time_point <= origin_track.last_time_point():
        return origin_track.find_position_at_time_point_number(time_point.time_point_number())

    if time_point < origin_track.first_time_point():
        # Go back in time to find previous tracks
        matched_positions = list()

        for previous_track in origin_track.find_all_previous_tracks(include_self=False):
            if previous_track.first_time_point() <= time_point <= previous_track.last_time_point():
                matched_positions.append(previous_track.find_position_at_time_point_number(time_point.time_point_number()))

        if len(matched_positions) == 0:
            return None

        avg_x = sum(pos.x for pos in matched_positions) / len(matched_positions)
        avg_y = sum(pos.y for pos in matched_positions) / len(matched_positions)
        avg_z = sum(pos.z for pos in matched_positions) / len(matched_positions)
        return Position(avg_x, avg_y, avg_z, time_point=time_point)

    # Go forward in time to find descending tracks
    matched_positions = list()
    for descending_track in origin_track.find_all_descending_tracks(include_self=False):
        if descending_track.first_time_point() <= time_point <= descending_track.last_time_point():
            matched_positions.append(descending_track.find_position_at_time_point_number(time_point.time_point_number()))

    if len(matched_positions) == 0:
        return None

    avg_x = sum(pos.x for pos in matched_positions) / len(matched_positions)
    avg_y = sum(pos.y for pos in matched_positions) / len(matched_positions)
    avg_z = sum(pos.z for pos in matched_positions) / len(matched_positions)
    return Position(avg_x, avg_y, avg_z, time_point=time_point)


class _ExportMovieJob(WorkerJob):

    _origin_position: Position
    _crop_size_xy_px: int
    _channel: ImageChannel
    _output_file: str

    def __init__(self, position: Position, crop_size_xy_px: int, channel: ImageChannel, output_file: str):
        self._origin_position = position
        self._crop_size_xy_px = crop_size_xy_px
        self._channel = channel
        self._output_file = output_file

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, positions=True, links=True, name=True)

    def gather_data(self, experiment_copy: Experiment) -> Any:
        origin_track = experiment_copy.links.get_track(self._origin_position)
        if origin_track is None:
            # Should have been caught earlier
            raise ValueError("The selected position is not part of any track.")

        min_time_point_number = self._origin_position.time_point_number()
        max_time_point_number = self._origin_position.time_point_number()
        for track in origin_track.find_all_previous_and_descending_tracks(include_self=True):
            min_time_point_number = min(min_time_point_number, track.first_time_point_number())
            max_time_point_number = max(max_time_point_number, track.last_time_point_number())

        time_point_count = max_time_point_number - min_time_point_number + 1
        array = None
        for i in range(time_point_count):
            time_point = TimePoint(min_time_point_number + i)
            average_position = _find_average_position(origin_track, time_point)
            if average_position is None:
                continue

            # Load the image slice
            image_2d = experiment_copy.images.get_image_slice_2d(time_point, self._channel, round(average_position.z))
            if image_2d is None:
                continue

            # Initialize the array on the first successful image load (now that we know the dtype)
            if array is None:
                array = numpy.zeros((time_point_count, self._crop_size_xy_px, self._crop_size_xy_px), dtype=image_2d.dtype)

            # Crop around the average position
            offset = experiment_copy.images.offsets.of_time_point(time_point)
            center_x = round(average_position.x - offset.x)
            center_y = round(average_position.y - offset.y)
            min_x = center_x - self._crop_size_xy_px // 2
            min_y = center_y - self._crop_size_xy_px // 2
            cropper.crop_2d(image_2d, min_x, min_y, array[i])

        return array

    def use_data(self, tab: SingleGuiTab, data: Any):
        tifffile.imwrite(self._output_file, data, imagej=True, metadata={'axes': 'TYX', 'Unit': 'frames'})

    def on_finished(self, data: Iterable[Any]):
        dialog.popup_message("Movie created", f"Done! The movie has been saved at {self._output_file}.")
