from typing import Tuple, List, Optional, Dict, Any

from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window


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
    image_series_loader_dialog.prompt_image_series(temporary_experiment)

    window.perform_data_action(_TimelapseAppendAction(experiment.images.image_loader(),
                                                      temporary_experiment.images.image_loader()))


def _prepend_timelapse(window: Window):
    experiment = window.get_experiment()
    if experiment.images.image_loader().first_time_point_number() is None:
        raise UserError("Cannot prepend images", "Cannot prepend images, as the starting time point of the"
                                                 " current image series is not defined.")

    temporary_experiment = Experiment()

    from organoid_tracker.gui import image_series_loader_dialog
    image_series_loader_dialog.prompt_image_series(temporary_experiment)
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
        appending_image_loader = _AppendingImageLoader([self._old_loader, self._appended_loader])
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
        appending_image_loader = _AppendingImageLoader([self._prepended_loader, self._old_loader])
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

class _AppendingImageLoader(ImageLoader):
    """Combines to image loaders, showing images after each other."""
    _internal: List[ImageLoader]

    def __init__(self, image_loaders: List[ImageLoader]):
        self._internal = list()
        for image_loader in image_loaders:
            image_loader = image_loader.uncached()
            if not image_loader.has_images():
                continue

            if isinstance(image_loader, _AppendingImageLoader):
                # Avoid double-wrapping, could hurt performance
                self._internal += image_loader._internal
            else:
                self._internal.append(image_loader)

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if len(self._internal) == 0:
            return None

        time_point_number = time_point.time_point_number()

        # List of channels is different for different image loaders, so use the channel index instead
        try:
            channel_index = self.get_channels().index(image_channel)
        except ValueError:
            return None  # Channel not available

        image_loader_index = 0
        while True:
            # Check channels
            channels = self._internal[image_loader_index].get_channels()
            if channel_index >= len(channels):
                return None  # Not that many channels available for this ImageLoader

            if time_point_number <= self._internal[image_loader_index].last_time_point_number():
                return self._internal[image_loader_index].get_3d_image_array(
                    TimePoint(time_point_number),
                    channels[channel_index])

            # Out of bounds for this time lapse, on to the next
            time_point_number -= self._internal[image_loader_index].last_time_point_number() + 1
            image_loader_index += 1
            if image_loader_index >= len(self._internal):
                return None  # Out of images
            time_point_number += self._internal[image_loader_index].first_time_point_number()

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if len(self._internal) == 0:
            return None

        time_point_number = time_point.time_point_number()

        image_loader_index = 0
        while True:
            if time_point_number <= self._internal[image_loader_index].last_time_point_number():
                return self._internal[image_loader_index].get_2d_image_array(
                    TimePoint(time_point_number), image_channel, image_z)

            # Out of bounds for this time lapse, on to the next
            time_point_number -= self._internal[image_loader_index].last_time_point_number() + 1
            image_loader_index += 1
            if image_loader_index >= len(self._internal):
                return None  # Out of images
            time_point_number += self._internal[image_loader_index].first_time_point_number()

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        if len(self._internal) == 0:
            return None
        return self._internal[0].get_image_size_zyx()

    def first_time_point_number(self) -> Optional[int]:
        if len(self._internal) == 0:
            return None
        return self._internal[0].first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        if len(self._internal) == 0:
            return None

        image_count = 0
        for internal in self._internal:
            new_last = internal.last_time_point_number()
            new_first = internal.first_time_point_number()
            if new_last is None or new_first is None:
                return None
            image_count += new_last - new_first + 1
        return image_count + self._internal[0].first_time_point_number() - 1

    def get_channel_count(self) -> int:
        channels = 0
        for internal in self._internal:
            channels = max(channels, internal.get_channel_count())
        return channels

    def serialize_to_config(self) -> Tuple[str, str]:
        if len(self._internal) == 0:
            return "", ""

        return self._internal[0].serialize_to_config()

    def copy(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._internal:
            new_internal.append(internal.copy())
        return _AppendingImageLoader(new_internal)

    def uncached(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._internal:
            new_internal.append(internal.uncached())
        return _AppendingImageLoader(new_internal)
