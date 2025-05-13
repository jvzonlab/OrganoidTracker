import math
from typing import Dict, Any, List, Optional

from organoid_tracker.core import bounding_box
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.gui_experiment import GuiExperiment, SingleGuiTab
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Other-Adjust Z-coords for maximal intensity...":
            lambda: _adjust_z_coords(window)
    }


def _adjust_z_coords(window: Window):
    radius_um = dialog.prompt_float("Adjusting Z-coords",
                                    "This function will move positions one Z-coord higher or lower, if that results"
                                    "\nin a higher intensity, as measured using a circle of a given radius. The intensity"
                                    "\nis measured in the current channel."
                                    ""
                                    "\n\nTo continue, enter a radius in micrometers and press OK.",
                                    minimum=0.1, maximum=100, default=3)
    if radius_um is None:
        return

    window.set_status("Started moving positions...")
    window.get_scheduler().add_task(_AdjustZCoordsTask(window, radius_um))


class _AdjustZCoordsTask(Task):
    _window: Window
    _processing_tabs: List[SingleGuiTab]
    _experiment_copies: List[Experiment]
    _radius_um: float
    _channel: ImageChannel

    def __init__(self, window: Window, radius_um: float):
        self._window = window
        self._processing_tabs = window.get_gui_experiment().get_active_tabs()
        self._experiment_copies = [tab.experiment.copy_selected(images=True, positions=True) for tab in
                                   self._processing_tabs]
        self._radius_um = radius_um
        self._channel = window.display_settings.image_channel

    def compute(self) -> List[Dict[Position, Position]]:
        results = list()
        for experiment in self._experiment_copies:
            moves = dict()

            circular_mask = _create_circular_mask(self._radius_um, experiment.images.resolution())
            for time_point in experiment.time_points():
                image = experiment.images.get_image(time_point, self._channel)
                for position in experiment.positions.of_time_point(time_point):

                    intensity = _get_intensity(position, image, circular_mask)
                    if intensity is None:
                        continue
                    intensity_up = _get_intensity(position.with_offset(0, 0, 1), image, circular_mask)
                    intensity_down = _get_intensity(position.with_offset(0, 0, -1), image, circular_mask)
                    if intensity_up is not None and intensity_up > intensity:
                        if intensity_down is not None and intensity_down > intensity_up:
                            # Intensity_down is the largest value, although intensity_up is also larger than intensity
                            moves[position] = position.with_offset(0, 0, -1)
                        else:
                            # Intensity_up is the largest value
                            moves[position] = position.with_offset(0, 0, 1)
                    elif intensity_down is not None and intensity_down > intensity:
                        # Intensity_down is the largest value
                        moves[position] = position.with_offset(0, 0, -1)
                    else:
                        # No adjustment needed
                        pass
            results.append(moves)
        return results

    def on_finished(self, result: List[Dict[Position, Position]]):
        total_positions = 0
        for tab, tab_result in zip(self._processing_tabs, result):
            total_positions += len(tab_result)
            tab.undo_redo.do(_MovePositionsAction(tab_result), tab.experiment)
        self._window.redraw_data()
        if len(self._processing_tabs) > 1:
            self._window.set_status(
                f"Adjusted {total_positions} positions across {len(self._processing_tabs)} experiments.")
        else:
            self._window.set_status(f"Adjusted {total_positions} positions.")


def _create_circular_mask(radius_um: float, resolution: ImageResolution) -> Mask:
    """Creates a mask that is circular in the xy plane."""
    radius_x_px = math.ceil(radius_um / resolution.pixel_size_x_um)
    radius_y_px = math.ceil(radius_um / resolution.pixel_size_y_um)
    mask = Mask(bounding_box.ONE.expanded(radius_x_px, radius_y_px, 0))

    # Evaluate the spheroid function to draw it
    mask.add_from_function(lambda x, y, z:
                           x ** 2 / radius_x_px ** 2 + y ** 2 / radius_y_px ** 2 <= 1)

    return mask


def _get_intensity(position: Position, intensity_image: Image, mask: Mask) -> Optional[int]:
    mask.center_around(position)
    if mask.count_pixels() == 0:
        return None
    masked_image = mask.create_masked_image(intensity_image)
    return int(masked_image.sum())


class _MovePositionsAction(UndoableAction):
    _position_moves: Dict[Position, Position]

    def __init__(self, position_moves: Dict[Position, Position]):
        self._position_moves = position_moves

    def do(self, experiment: Experiment) -> str:
        for from_position, to_position in self._position_moves.items():
            experiment.move_position(from_position, to_position)
        return f"Adjusted {len(self._position_moves)} positions"

    def undo(self, experiment: Experiment) -> str:
        for from_position, to_position in self._position_moves.items():
            experiment.move_position(to_position, from_position)
        return f"Moved {len(self._position_moves)} positions back"
