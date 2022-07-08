from typing import Tuple, List, Optional, Dict, Any

from numpy import ndarray

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Batch-Append image channel...": lambda: _append_channel(window)
    }


def _append_channel(window: Window):
    experiment = window.get_experiment()
    temporary_experiment = Experiment()

    from organoid_tracker.gui import image_series_loader_dialog
    image_series_loader_dialog.prompt_image_series(temporary_experiment)

    window.perform_data_action(_ChannelAppendAction(experiment.images.image_loader(),
                                                    temporary_experiment.images.image_loader()))


class _ChannelAppendAction(UndoableAction):

    _old_loader: ImageLoader
    _appended_loader: ImageLoader

    def __init__(self, old_loader: ImageLoader, appended_loader: ImageLoader):
        self._old_loader = old_loader
        self._appended_loader = appended_loader

    def do(self, experiment: Experiment) -> str:
        appending_image_loader = _ChannelAppendingImageLoader([self._old_loader, self._appended_loader])
        experiment.images.image_loader(appending_image_loader)
        return f"Appended the channels. We now have {len(appending_image_loader.get_channels())} channels available."

    def undo(self, experiment: Experiment) -> str:
        experiment.images.image_loader(self._old_loader)
        return f"Removed the appended channels again"


class _ChannelAppendingImageLoader(ImageLoader):
    """Combines to image loaders, showing images after each other."""

    # A list of all image loaders, where every loader appears once in the list
    _unique_loaders: List[ImageLoader]

    def __init__(self, image_loaders: List[ImageLoader]):
        self._unique_loaders = list()

        for image_loader in image_loaders:
            if not image_loader.has_images():
                continue

            if isinstance(image_loader, _ChannelAppendingImageLoader):
                # Avoid double-wrapping, could hurt performance
                self._unique_loaders += image_loader._unique_loaders
            else:
                self._unique_loaders.append(image_loader)

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        # Iterate over all image loaders
        image_channel_index = image_channel.index_zero
        for image_loader in self._unique_loaders:
            if image_channel_index < image_loader.get_channel_count():
                return image_loader.get_3d_image_array(time_point, ImageChannel(index_zero=image_channel_index))
            image_channel_index -= image_loader.get_channel_count()

        return None

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        # Iterate over all image loaders
        image_channel_index = image_channel.index_zero
        for image_loader in self._unique_loaders:
            if image_channel_index < image_loader.get_channel_count():
                return image_loader.get_2d_image_array(time_point, ImageChannel(index_zero=image_channel_index), image_z)
            image_channel_index -= image_loader.get_channel_count()

        return None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        # Returns the size only if all image loaders have the same image size
        the_size = None
        for image_loader in self._unique_loaders:
            new_size = image_loader.get_image_size_zyx()
            if new_size is None:
                return None  # Not all sizes are known, so return None
            if the_size is None:
                the_size = new_size  # Set the first size
            else:
                if the_size != new_size:
                    return None  # Different channels have different sizes, so return None
        return the_size

    def first_time_point_number(self) -> Optional[int]:
        # Return the first time point number for which we have at least one channel
        first_time_point_numbers = [image_loader.first_time_point_number() for image_loader in self._unique_loaders]
        return min_none(first_time_point_numbers)

    def last_time_point_number(self) -> Optional[int]:
        # Return the last time point number for which we have at least one channel
        last_time_point_numbers = [image_loader.last_time_point_number() for image_loader in self._unique_loaders]
        return max_none(last_time_point_numbers)

    def get_channel_count(self) -> int:
        count = 0
        for image_loader in self._unique_loaders:
            count += image_loader.get_channel_count()
        return count

    def serialize_to_config(self) -> Tuple[str, str]:
        for image_loader in self._unique_loaders:
            return image_loader.serialize_to_config()
        return "", ""

    def copy(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._unique_loaders:
            new_internal.append(internal.copy())
        return _ChannelAppendingImageLoader(new_internal)

    def uncached(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._unique_loaders:
            new_internal.append(internal.uncached())
        return _ChannelAppendingImageLoader(new_internal)
