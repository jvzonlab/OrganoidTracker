from typing import Dict, Any

from organoid_tracker.core import UserError
from organoid_tracker.core.images import Images
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.image_loading import builtin_image_filters
from organoid_tracker.image_loading.builtin_image_filters import GaussianBlurFilter, MultiplyPixelsFilter, \
    ThresholdFilter
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "View//Image-Image filters//Filter-Increase brightness...": lambda: _enhance_brightness(window),
        "View//Image-Image filters//Filter-Set min-max intensity...": lambda: _set_min_max_intensity(window),
        "View//Image-Image filters//Filter-Threshold...": lambda: _threshold(window),
        "View//Image-Image filters//Filter-Gaussian blur...": lambda: _gaussian_blur(window),
        "View//Image-Image filters//Filter-Merge channels...": lambda: _merge_channels(window),
        "View//Image-Image filters//Remove-Remove all filters": lambda: _remove_filters(window)
    }


def _threshold(window: Window):
    min_value = dialog.prompt_float("Threshold", "What is the threshold for suppressing noise? (0% - 100%)"
                                    "\n\nA value of 25 removes all pixels with a value less than 25% of the maximum"
                                    " brightness in the image.", minimum=0, maximum=100, default=8)
    if min_value is None:
        return

    image_channel = window.display_settings.image_channel
    for experiment in window.get_active_experiments():
        experiment.images.filters.add_filter(image_channel, ThresholdFilter(min_value / 100))
    window.get_gui_experiment().redraw_image_and_data()


def _gaussian_blur(window: Window):
    value = dialog.prompt_float("Blur radius", "What is the blur radius in pixels?",
                                minimum=0.1, maximum=31, default=5)
    if value is None:
        return

    image_channel = window.display_settings.image_channel
    for experiment in window.get_active_experiments():
        experiment.images.filters.add_filter(image_channel, GaussianBlurFilter(value))
    window.get_gui_experiment().redraw_image_and_data()


def _enhance_brightness(window: Window):
    multiplier = dialog.prompt_float("Multiplier", "How many times would you like to increase the brightness of the"
                                     " image?", minimum=1, maximum=100)
    if multiplier is None:
        return

    image_channel = window.display_settings.image_channel
    for experiment in window.get_active_experiments():
        experiment.images.filters.add_filter(image_channel, MultiplyPixelsFilter(multiplier))
    window.get_gui_experiment().redraw_image_and_data()


def _set_min_max_intensity(window: Window):
    min_intensity = dialog.prompt_float("Min intensity", "What should the minimum intensity be?", minimum=0, maximum=1e9)
    if min_intensity is None:
        return
    max_intensity = dialog.prompt_float("Max intensity", "What should the maximum intensity be?", minimum=min_intensity, maximum=1e9)
    if max_intensity is None:
        return

    image_channel = window.display_settings.image_channel
    for experiment in window.get_active_experiments():
        experiment.images.filters.add_filter(image_channel, builtin_image_filters.create_min_max_filter(min_intensity, max_intensity))
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

    channel_merger = ChannelSummingImageLoader(images.image_loader(), channel_groups)
    images.image_loader(channel_merger)
    window.get_gui_experiment().redraw_image_and_data()


def _remove_filters(window: Window):
    removed_count = 0
    experiment_count = 0
    image_channel = window.display_settings.image_channel

    for experiment in window.get_active_experiments():
        experiment_count += 1
        images = experiment.images

        # Undo channel merging
        image_loader = images.image_loader()
        if isinstance(image_loader, ChannelSummingImageLoader):
            images.image_loader(image_loader.get_unmerged_image_loader())
            removed_count += 1

        # Undo filters
        filters = list(images.filters.of_channel(image_channel))
        removed_count += len(filters)
        images.filters.clear_channel(image_channel)

    if experiment_count == 1:
        if removed_count == 1:
            window.set_status(f"Removed 1 filter for channel {image_channel.index_one}.")
        else:
            window.set_status(f"Removed {removed_count} filters for channel {image_channel.index_one}.")
    else:
        if removed_count == 1:
            window.set_status(f"Removed 1 filter in total across all tabs for channel {image_channel.index_one}.")
        else:
            window.set_status(f"Removed {removed_count} filters in total across for channel {image_channel.index_one}.")

    window.get_gui_experiment().redraw_image_and_data()

