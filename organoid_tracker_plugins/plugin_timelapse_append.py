from typing import Dict, Any

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.image_loading.builtin_merging_image_loaders import TimeAppendingImageLoader


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Batch-Prepend image series...": lambda: _prepend_timelapse(window),
        "Edit//Batch-Append image series...": lambda: _append_timelapse(window)
    }


def _append_timelapse(window: Window):
    experiment = window.get_experiment()
    temporary_experiment = Experiment()
    if experiment.images.image_loader().last_time_point_number() is None:
        raise UserError("Cannot append images", "Cannot append images, as the ending time point of the"
                                                 " current image series is not defined.")

    from organoid_tracker.gui import image_series_loader_dialog
    if not image_series_loader_dialog.prompt_image_series(window.registry.get_registered_file_loaders(), temporary_experiment):
        return

    window.perform_data_action(_TimelapseAppendAction(experiment.images.image_loader(),
                                                      temporary_experiment.images.image_loader()))


def _prepend_timelapse(window: Window):
    experiment = window.get_experiment()
    if experiment.images.image_loader().first_time_point_number() is None:
        raise UserError("Cannot prepend images", "Cannot prepend images, as the starting time point of the"
                                                 " current image series is not defined.")

    temporary_experiment = Experiment()

    from organoid_tracker.gui import image_series_loader_dialog
    if not image_series_loader_dialog.prompt_image_series(window.registry.get_registered_file_loaders(), temporary_experiment):
        return
    prepending_loader = temporary_experiment.images.image_loader()
    if prepending_loader.last_time_point_number() is None:
        raise UserError("Cannot prepend images", "Cannot prepend images, as the ending time point of the"
                                                 " selected image series is not defined.")

    window.perform_data_action(_TimelapsePrependAction(old_loader=experiment.images.image_loader(),
                                                       prepending_loader=temporary_experiment.images.image_loader()))


class _TimelapseAppendAction(UndoableAction):

    _old_loader: ImageLoader
    _appended_loader: ImageLoader

    def __init__(self, old_loader: ImageLoader, appended_loader: ImageLoader):
        self._old_loader = old_loader
        self._appended_loader = appended_loader

    def do(self, experiment: Experiment) -> str:
        appending_image_loader = TimeAppendingImageLoader([self._old_loader, self._appended_loader])
        experiment.images.image_loader(appending_image_loader)
        return f"Appended the time. Images now run up until time point" \
               f" {appending_image_loader.last_time_point_number()}."

    def undo(self, experiment: Experiment) -> str:
        experiment.images.image_loader(self._old_loader)
        return f"Removed the appended images again"


class _TimelapsePrependAction(UndoableAction):

    _old_loader: ImageLoader
    _prepended_loader: ImageLoader

    def __init__(self, *, old_loader: ImageLoader, prepending_loader: ImageLoader):
        self._old_loader = old_loader
        self._prepended_loader = prepending_loader

    def do(self, experiment: Experiment) -> str:
        # Say: previously the experiment started at time point 1, and the prepended loader runs from time point 0 to 8
        # Then the experiment now needs to start at time point 9, so an offset of 8

        # Move all tracking data
        new_start = self._prepended_loader.last_time_point_number() + 1
        old_start = self._old_loader.first_time_point_number()
        offset = new_start - old_start
        experiment.move_in_time(offset)

        # Inject new image loader
        appending_image_loader = TimeAppendingImageLoader([self._prepended_loader, self._old_loader])
        experiment.images.image_loader(appending_image_loader)

        return f"Prepended the images. We now have {offset} new image(s) at the start."

    def undo(self, experiment: Experiment) -> str:
        # Move all tracking data back
        new_start = self._prepended_loader.last_time_point_number() + 1
        old_start = self._old_loader.first_time_point_number()
        offset = new_start - old_start
        experiment.move_in_time(-offset)

        experiment.images.image_loader(self._old_loader)
        return f"Removed the prepended images again"
