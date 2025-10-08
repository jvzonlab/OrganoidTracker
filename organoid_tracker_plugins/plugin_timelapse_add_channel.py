from typing import Tuple, List, Optional, Dict, Any

from numpy import ndarray

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelAppendingImageLoader


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Batch-Append image channel...": lambda: _append_channel(window)
    }


def _append_channel(window: Window):
    experiment = window.get_experiment()
    temporary_experiment = Experiment()

    from organoid_tracker.gui import image_series_loader_dialog
    image_series_loader_dialog.prompt_image_series(window.registry.get_registered_file_loaders(), temporary_experiment)

    window.perform_data_action(_ChannelAppendAction(experiment.images.image_loader(),
                                                    temporary_experiment.images.image_loader()))


class _ChannelAppendAction(UndoableAction):

    _old_loader: ImageLoader
    _appended_loader: ImageLoader

    def __init__(self, old_loader: ImageLoader, appended_loader: ImageLoader):
        self._old_loader = old_loader
        self._appended_loader = appended_loader

        self.needs_saving = False  # This action won't make any changes to the AUT file

    def do(self, experiment: Experiment) -> str:
        appending_image_loader = ChannelAppendingImageLoader([self._old_loader, self._appended_loader])
        experiment.images.image_loader(appending_image_loader)
        return f"Appended the channels. We now have {len(appending_image_loader.get_channels())} channels available."

    def undo(self, experiment: Experiment) -> str:
        experiment.images.image_loader(self._old_loader)
        return f"Removed the appended channels again"

