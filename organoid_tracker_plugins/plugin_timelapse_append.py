from typing import Dict, Any

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.image_loading.builtin_merging_image_loaders import TimeAppendingImageLoader


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Batch-Append image series at end...": lambda: _append_timelapse(window)
    }


def _append_timelapse(window: Window):
    experiment = window.get_experiment()
    temporary_experiment = Experiment()

    from organoid_tracker.gui import image_series_loader_dialog
    image_series_loader_dialog.prompt_image_series(temporary_experiment)

    window.perform_data_action(_TimelapseAppendAction(experiment.images.image_loader(),
                                                      temporary_experiment.images.image_loader()))


class _TimelapseAppendAction(UndoableAction):

    _old_loader: ImageLoader
    _appended_loader: ImageLoader

    def __init__(self, old_loader: ImageLoader, appended_loader: ImageLoader):
        self._old_loader = old_loader
        self._appended_loader = appended_loader

        self.needs_saving = False  # This action won't make any changes to the AUT file

    def do(self, experiment: Experiment) -> str:
        appending_image_loader = TimeAppendingImageLoader([self._old_loader, self._appended_loader])
        experiment.images.image_loader(appending_image_loader)
        return f"Appended the time. Images now run up until time point" \
               f" {appending_image_loader.last_time_point_number()}."

    def undo(self, experiment: Experiment) -> str:
        experiment.images.image_loader(self._old_loader)
        return f"Removed the appended images again"
