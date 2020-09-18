"""Deletes all positions more than X um away from any nucleus center in some other dataset."""
from typing import Dict, Any, List

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import io
from organoid_tracker.linking.nearby_position_finder import find_close_positions


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Batch-Prune to other dataset...": lambda: _prune_positions_start(window)
    }

def _prune_positions_start(window: Window):
    current_experiment = window.get_experiment()
    distance_um = dialog.prompt_float("Pruning", "This command will prune the current dataset so that all positions\n"
                                                 "more than X µm from another dataset is gone. First, we will ask to\n"
                                                 "define that distance, in micrometers.", minimum=0, default=5)
    if distance_um is None:
        return
    if not dialog.prompt_confirmation("Pruning", "Next, we will ask you to select the file of the\n"
                                                 "other dataset. This dataset is not modified."):
        return
    file_name = dialog.prompt_load_file("Select data file", io.SUPPORTED_IMPORT_FILES)
    if file_name is None:
        return  # Cancelled
    window.get_scheduler().add_task(_PruneTask(window.get_gui_experiment(), file_name, distance_um))



class _PruneTask(Task):

    # Only access these two from the main thread!
    _experiment_gui: GuiExperiment
    _experiment: Experiment

    _positions_copy: PositionCollection
    _resolution: ImageResolution
    _prune_mask_file: str
    _max_distance_um: float

    def __init__(self, experiment_gui: GuiExperiment, prune_mask_file: str, max_distance_um: float):
        self._experiment_gui = experiment_gui
        self._experiment = experiment_gui.get_experiment()
        self._positions_copy = self._experiment.positions.copy()
        self._resolution = self._experiment.images.resolution()
        self._prune_mask_file = prune_mask_file
        self._max_distance_um = max_distance_um

    def compute(self) -> List[Position]:
        to_delete = list()

        prune_mask = io.load_data_file(self._prune_mask_file)
        for time_point in self._positions_copy.time_points():
            scratch_positions = prune_mask.positions.of_time_point(time_point)
            for position in self._positions_copy.of_time_point(time_point):
                nearest_in_scratch = find_close_positions(scratch_positions, around=position, tolerance=1,
                                                          resolution=self._resolution, max_distance_um=self._max_distance_um)
                if len(nearest_in_scratch) == 0:
                    # No nearby positions in scratch, delete this position
                    to_delete.append(position)
        return to_delete

    def on_finished(self, result: List[Position]):
        self._experiment.remove_positions(result)
        self._experiment_gui.redraw_data()
        dialog.popup_message("Pruning completed", f"Pruning completed: all {len(result)} positions more than"
                                                  f" {self._max_distance_um:.1f} µm from any position in the other"
                                                  f" dataset are now removed.")
