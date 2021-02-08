from typing import Dict, Any

import numpy
from numpy.core.multiarray import ndarray

from organoid_tracker.core import UserError
from organoid_tracker.core.image_loader import ImageFilter
from organoid_tracker.core.images import Images
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.image_loading.channel_merging_image_loader import ChannelMergingImageLoader
from organoid_tracker.image_loading.noise_suppressing_filters import ThresholdFilter, GaussianBlurFilter


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "View//Image-Image filters//Increase brightness...": lambda: _enhance_brightness(window),
        "View//Image-Image filters//Threshold...": lambda: _threshold(window),
        "View//Image-Image filters//Gaussian blur...": lambda: _gaussian_blur(window),
        "View//Image-Image filters//Merge channels...": lambda: _merge_channels(window),
        "View//Image-Image filters//Remove-Remove all filters": lambda: _remove_filters(window)
    }


def _threshold(window: Window):
    min_value = dialog.prompt_float("Threshold", "What is the threshold for suppressing noise? (0% - 100%)"
                                    "\n\nA value of 25 removes all pixels with a value less than 25% of the maximum"
                                    " brightness in the image.", minimum=0, maximum=100, default=8)
    if min_value is None:
        return

    window.get_experiment().images.filters.append(ThresholdFilter(min_value / 100))
    window.get_gui_experiment().redraw_image_and_data()


def _gaussian_blur(window: Window):
    value = dialog.prompt_int("Blur radius", "What is the blur radius in pixels? Only odd numbers are allowed.",
                              minimum=1, maximum=31, default=5)
    if value is None:
        return
    if value % 2 == 0:
        raise UserError("Even number", f"Cannot use the even number {value} - the blur radius must be an odd number.")

    window.get_experiment().images.filters.append(GaussianBlurFilter(value))
    window.get_gui_experiment().redraw_image_and_data()


def _enhance_brightness(window: Window):
    multiplier = dialog.prompt_float("Multiplier", "How many times would you like to increase the brightness of the"
                                     " image?", minimum=1, maximum=100)
    if multiplier is None:
        return

    window.get_experiment().images.filters.append(_IncreaseBrightnessFilter(multiplier))
    window.get_gui_experiment().redraw_image_and_data()


def _merge_channels(window: Window):
    """Not implemented as a filter, but as a separate image loader."""
    images: Images = window.get_experiment().images
    channels = images.get_channels()
    channel_ids = list(range(len(channels)))
    channel_names = ["Channel " + str(i + 1) for i in channel_ids]

    chosen_channel_ids = option_choose_dialog.prompt_list_multiple("Channels to merge",
                                                                   "Which channels should be merged?",
                                                                   "Channels:", channel_names)
    if chosen_channel_ids is None:
        return
    if len(chosen_channel_ids) == 0:
        raise UserError("No channels selected", "No channels were selected.")
    remaining_channel_ids = [i for i in channel_ids if i not in chosen_channel_ids]

    # Build the new channel groups
    channel_groups = list()
    channel_groups.append([channels[i] for i in chosen_channel_ids])  # Merge
    for remaining_channel_id in remaining_channel_ids:
        # Keep these separate
        channel_groups.append([channels[remaining_channel_id]])

    channel_merger = ChannelMergingImageLoader(images.image_loader(), channel_groups)
    images.image_loader(channel_merger)
    window.get_gui_experiment().redraw_image_and_data()


def _remove_filters(window: Window):
    images = window.get_experiment().images

    removed_count = 0

    # Undo channel merging
    image_loader = images.image_loader()
    if isinstance(image_loader, ChannelMergingImageLoader):
        images.image_loader(image_loader.get_unmerged_image_loader())
        removed_count += 1

    # Undo filters
    filters = images.filters
    removed_count += len(filters)
    filters.clear()

    if removed_count == 1:
        window.set_status("Removed 1 filter.")
    else:
        window.set_status(f"Removed {removed_count} filters.")

    window.get_gui_experiment().redraw_image_and_data()


class _IncreaseBrightnessFilter(ImageFilter):
    _factor: float

    def __init__(self, factor: float):
        if factor < 0:
            raise ValueError("factor may not be negative, but was " + str(factor))
        self._factor = factor

    def filter(self, image_8bit: ndarray):
        if int(self._factor) == self._factor:
            # Easy, apply cheap integer multiplication - costs almost no RAM

            # First get rid of things that will overflow
            new_max = int(255 / self._factor)
            image_8bit[image_8bit > new_max] = new_max

            # Then do integer multiplication
            image_8bit *= int(self._factor)
            return

        # Copying required
        scaled = image_8bit * self._factor
        scaled[scaled > 255] = 255  # Prevent overflow
        image_8bit[...] = scaled.astype(numpy.uint8)

    def copy(self):
        return _IncreaseBrightnessFilter(self._factor)

    def get_name(self) -> str:
        return "Increase brightness"
