import random
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Dict, Any, Tuple, List, Set

import matplotlib.cm
import numpy
import scipy
import skimage.measure
from matplotlib.colors import Colormap, ListedColormap
from numpy import ndarray

from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class _MaskProcessingMode(ABC):

    @abstractmethod
    def get_name(self) -> str:
        """Gets the name of this measurement mode."""
        raise NotImplementedError()

    @abstractmethod
    def get_size_question(self) -> Optional[str]:
        """The process_mask method requires a size parameter. This method returns the question that we should ask the
        user so that the user knows what value to choose for that parameter. Like "By how many pixels should we enlarge
        the mask?". Returns None if a size parameter is not required (in which case you can pass 0 to process_mask)."""
        raise NotImplementedError()


    def process_mask_3d(self, labeled_image_3d: ndarray, size: int) -> ndarray:
        """Calls process_mask_2d layer by layer."""
        processed = numpy.empty_like(labeled_image_3d)
        for z in range(labeled_image_3d.shape[0]):
            processed[z] = self.process_mask_2d(labeled_image_3d[z], size)
        return processed

    @abstractmethod
    def process_mask_2d(self, labeled_image_2d: ndarray, size: int):
        """Returns a mask that is used for the measurements. The new mask uses the same labeling as the old mask. The
                old mask is not modified. See self.get_size_question for the size parameter."""
        raise NotImplementedError()


class _ModeInside(_MaskProcessingMode):
    def get_name(self) -> str:
        return "Inside masks (default)"

    def get_size_question(self) -> Optional[str]:
        return None

    def process_mask_3d(self, labeled_image_3d: ndarray, size: int) -> ndarray:
        return labeled_image_3d  # Don't change

    def process_mask_2d(self, labeled_image_2d: ndarray, size: int):
        return labeled_image_2d  # Don't change


class _ModeShrunken(_MaskProcessingMode):
    def get_name(self) -> str:
        return "In shrunken masks"

    def get_size_question(self) -> Optional[str]:
        return "By how many pixels should we shrink the mask (in 2D)?"

    def process_mask_2d(self, labeled_image_2d: ndarray, size: int) -> ndarray:
        eroded = scipy.ndimage.binary_erosion(labeled_image_2d, iterations=size)
        eroded_labeled = numpy.copy(labeled_image_2d)
        eroded_labeled[eroded == 0] = 0
        return eroded_labeled


class _ModeEnlarged(_MaskProcessingMode):
    def get_name(self) -> str:
        return "In enlarged masks"

    def get_size_question(self) -> Optional[str]:
        return "By how many pixels should we enlarge each mask (in 2D)?"

    def process_mask_2d(self, labeled_image_2d: ndarray, size: int) -> ndarray:
        expanded = scipy.ndimage.maximum_filter(labeled_image_2d, size=size)
        expanded[labeled_image_2d != 0] = labeled_image_2d[labeled_image_2d != 0]  # Leave original labels intact
        return expanded


class _ModeOutside(_MaskProcessingMode):
    def get_name(self) -> str:
        return "At the borders, on the outside"

    def get_size_question(self) -> Optional[str]:
        return "How many pixels should we measure outside the mask?"

    def process_mask_2d(self, labeled_image_2d: ndarray, size: int) -> ndarray:
        expanded = scipy.ndimage.maximum_filter(labeled_image_2d, size=size * 2 + 1)
        expanded[labeled_image_2d != 0] = 0  # Don't measure in original nuclei
        return expanded


_PROCESSING_MODES = [_ModeInside(), _ModeOutside(), _ModeShrunken(), _ModeEnlarged()]
_DEFAULT_PROCESSING_MODE = _ModeInside()


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Record intensities//Record-Record using pre-existing segmentation...": lambda: _view_intensities(window)
    }


def _view_intensities(window: Window):
    activate(_PreexistingSegmentationVisualizer(window))


def _by_label(region_props: List["skimage.measure._regionprops.RegionProperties"]
              ) -> Dict[int, "skimage.measure._regionprops.RegionProperties"]:
    return_value = dict()
    for region in region_props:
        return_value[region.label] = region
    return return_value


class _RecordIntensitiesJob(WorkerJob):
    """Records the intensities of all positions."""

    _segmentation_channel: ImageChannel
    _measurement_channel_1: ImageChannel
    _measurement_channel_2: Optional[ImageChannel]
    _mask_processing_mode: _MaskProcessingMode
    _mask_processing_size: int
    _intensity_key: str

    def __init__(self, segmentation_channel: ImageChannel, measurement_channel_1: ImageChannel,
                 measurement_channel_2: Optional[ImageChannel], *, mask_processing_mode: _MaskProcessingMode,
                 mask_processing_size: int, intensity_key: str):
        self._segmentation_channel = segmentation_channel
        self._measurement_channel_1 = measurement_channel_1
        self._measurement_channel_2 = measurement_channel_2
        self._mask_processing_mode = mask_processing_mode
        self._mask_processing_size = mask_processing_size
        self._intensity_key = intensity_key

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True, positions=True)

    def gather_data(self, experiment_copy: Experiment) -> Tuple[Dict[Position, float], Dict[Position, int]]:
        intensities = dict()
        volumes_px3 = dict()
        for time_point in experiment_copy.positions.time_points():
            print(f"Working on time point {time_point.time_point_number()}...")

            # Load images
            label_image = experiment_copy.images.get_image(time_point, self._segmentation_channel)
            measurement_image_1 = experiment_copy.images.get_image(time_point, self._measurement_channel_1)
            measurement_image_2 = None
            if self._measurement_channel_2 is not None:
                measurement_image_2 = experiment_copy.images.get_image(time_point, self._measurement_channel_2)
                if measurement_image_2 is None:
                    continue  # Skip this time point, an image is missing

            if label_image is None or measurement_image_1 is None:
                continue  # Skip this time point, an image is missing

            # Calculate intensities
            processed_labels = self._mask_processing_mode.process_mask_3d(label_image.array, self._mask_processing_size)
            props_by_label = _by_label(skimage.measure.regionprops(processed_labels))
            for position in experiment_copy.positions.of_time_point(time_point):
                index = label_image.value_at(position)
                if index == 0:
                    continue
                props = props_by_label.get(index)
                if props is None:
                    continue
                intensity = numpy.sum(measurement_image_1.array[props.slice] * props.image)
                if measurement_image_2 is not None:
                    intensity_2 = numpy.sum(measurement_image_2.array[props.slice] * props.image)
                    intensity /= intensity_2
                intensities[position] = float(intensity)
                volumes_px3[position] = props.area
        return intensities, volumes_px3

    def use_data(self, tab: SingleGuiTab, data: Tuple[Dict[Position, float], Dict[Position, int]]):
        intensities, volume_px3 = data
        intensity_calculator.set_raw_intensities(tab.experiment, intensities, volume_px3,
                                                 intensity_key=self._intensity_key)
        tab.undo_redo.mark_unsaved_changes()

    def on_finished(self, results: Any):
        dialog.popup_message("Intensities recorded", "All intensities have been recorded.\n\n"
                                                     "Your next step is likely to set a normalization. This can be\n"
                                                     "done from the Intensity menu in the main screen of the program.")


class _PreexistingSegmentationVisualizer(ExitableImageVisualizer):
    """First, specify the segmentation channel (containing a pre-segmented image) and the measurement channel in the
    Parameters menu. Then, use Edit -> Record intensities to record the average intensities.

    If you don't have pre-segmented images loaded yet, exit this view and use Edit -> Append image channel.
    """
    _segmented_channel: Optional[ImageChannel] = None
    _channel_1: Optional[ImageChannel] = None
    _channel_2: Optional[ImageChannel] = None
    _intensity_key: str = intensity_calculator.DEFAULT_INTENSITY_KEY
    _label_colormap: Colormap
    _mask_processing_mode: _MaskProcessingMode = _DEFAULT_PROCESSING_MODE
    _mask_processing_size: int = 0

    def __init__(self, window: Window):
        super().__init__(window)
        self._display_settings.max_intensity_projection = False

        # Initialize or random colormap
        source_colormap: Colormap = matplotlib.cm.jet
        samples = [source_colormap(sample_pos / 1000) for sample_pos in range(1000)]
        random.Random("fixed seed to ensure same colors").shuffle(samples)
        samples[0] = (0, 0, 0, 0)  # Force background to black
        samples[1] = (0, 0, 0, 0)  # Force first label to black too, this is also background
        self._label_colormap = ListedColormap(samples)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        options = {
            **super().get_extra_menu_options(),
            "Edit//Channels-Record intensities...": self._record_intensities,
            "Parameters//Channel-Set first measurement channel...": self._set_measurement_channel_one,
            "Parameters//Channel-Set second measurement channel (optional)...": self._set_measurement_channel_two,
            "Parameters//Other-Set segmented channel...": self._set_segmented_channel,
            "Parameters//Other-Set storage key...": self._set_intensity_key,
        }
        for mode in _PROCESSING_MODES:
            options["Parameters//Other-Set measurement location//" + mode.get_name()] =\
                partial(self._set_processing_mode, mode)
        return options

    def _set_processing_mode(self, mode: _MaskProcessingMode):
        """Changes self._mask_processing_size and self._mask_processing_mode"""
        new_size = 0
        question = mode.get_size_question()
        if question is not None:
            new_size = dialog.prompt_int("Set size", question, minimum=0, maximum=1000,
                                         default=self._mask_processing_size)
            if new_size is None:
                return

        self._mask_processing_size = new_size
        self._mask_processing_mode = mode
        self.refresh_data()
        self._window.set_status("Set the mask processing mode to \"" + mode.get_name() + "\".")

    def _set_intensity_key(self):
        """Prompts the user for a new intensity key."""
        new_key = dialog.prompt_str("Storage key",
                                    "Under what key should the intensities be stored?"
                                    "\nYou can choose a different value than the default if you want"
                                    " to maintain different sets of intensities.",
                                    default=self._intensity_key)
        if new_key is not None and len(new_key) > 0:
            self._intensity_key = new_key

    def _set_segmented_channel(self):
        """Prompts the user for a new value of self._segmentation_channel."""
        channels = self._experiment.images.get_channels()
        current_channel = self._segmented_channel if self._segmented_channel is not None else self._display_settings.image_channel
        channel_count = len(channels)

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._segmented_channel = channels[new_channel_index - 1]
            self.refresh_data()

    def _set_measurement_channel_one(self):
        """Prompts the user for a new value of self._channel1."""
        current_channel = self._channel_1 if self._channel_1 is not None else self._display_settings.image_channel
        channel_count = len(self._find_available_channels())

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use"
                                                                  f" (1-{channel_count}, inclusive)?", minimum=1,
                                              maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            self._channel_1 = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _set_measurement_channel_two(self):
        """Prompts the user for a new value of either self._channel2.
        """
        current_channel = self._channel_2 if self._channel_2 is not None else self._display_settings.image_channel
        channel_count = len(self._find_available_channels())

        new_channel_index = dialog.prompt_int("Select a channel", f"What channel do you want to use as the denominator"
                                                                  f" (1-{channel_count}, inclusive)?\n\nIf you don't want to compare two"
                                                                  f" channels, and just want to\nview one channel, set this value to 0.",
                                              minimum=0, maximum=channel_count,
                                              default=current_channel.index_one)
        if new_channel_index is not None:
            if new_channel_index == 0:
                self._channel_2 = None
            else:
                self._channel_2 = ImageChannel(index_zero=new_channel_index - 1)
            self.refresh_data()

    def _find_available_channels(self) -> Set[ImageChannel]:
        """Finds all channels that are available in all open experiments."""
        channels = set()
        for experiment in self._window.get_active_experiments():
            for channel in experiment.images.get_channels():
                channels.add(channel)
        return channels

    def _intensity_key_already_exists(self, key: str) -> bool:
        for experiment in self._window.get_active_experiments():
            if experiment.position_data.has_position_data_with_name(key):
                return True
        return False

    def _record_intensities(self):
        channels = self._find_available_channels()
        if self._segmented_channel is None or self._segmented_channel not in channels:
            raise UserError("Invalid segmentation channel", "Please set a segmentation channel in the Parameters menu.")
        if self._channel_1 is None or self._channel_1 not in channels:
            raise UserError("Invalid first channel", "Please set a measurement channel to measure in"
                                                     " using the Parameters menu.")
        if self._channel_2 is not None and self._channel_2 not in channels:
            raise UserError("Invalid second channel", "The selected second measurement channel is no longer available."
                                                      " Please select a new one in the Parameters menu.")
        if self._intensity_key_already_exists(self._intensity_key):
            if not dialog.prompt_confirmation("Intensities", "Warning: previous intensities stored under the key "
                                                             "\""+self._intensity_key+"\" will be overwritten.\n\n"
                                                             "This cannot be undone. Do you want to continue?\n\n"
                                                             "If you press Cancel, you can go back and choose a"
                                                             " different key in the Parameters menu."):
                return

        worker_job.submit_job(self._window,
            _RecordIntensitiesJob(self._segmented_channel, self._channel_1, self._channel_2,
                                  intensity_key=self._intensity_key, mask_processing_mode=self._mask_processing_mode,
                                  mask_processing_size=self._mask_processing_size))
        self.update_status("Started recording all intensities...")

    def should_show_image_reconstruction(self) -> bool:
        if self._segmented_channel is None:
            return False  # Nothing to draw
        if self._display_settings.image_channel not in {self._channel_1, self._channel_2, self._segmented_channel}:
            return False  # Nothing to draw for this channel
        return True

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._segmented_channel is None:
            return   # Nothing to draw
        if self._display_settings.image_channel not in {self._channel_1, self._channel_2, self._segmented_channel}:
            return  # Nothing to draw for this channel
        if self._segmented_channel == self._display_settings.image_channel:
            # Avoid drawing on top of the same image
            rgb_canvas_2d[:, :, 0:3] = 0
            if rgb_canvas_2d.shape[-1] == 4:
                rgb_canvas_2d[:, :, 3] = 1  # Also erase alpha channel

        labels = self._experiment.images.get_image_slice_2d(time_point, self._segmented_channel, z)
        if labels is None:
            return  # No image here
        labels = self._mask_processing_mode.process_mask_2d(labels, size=self._mask_processing_size)

        colored: ndarray = self._label_colormap(labels.flatten())
        colored = colored.reshape((rgb_canvas_2d.shape[0], rgb_canvas_2d.shape[1], 4))
        rgb_canvas_2d[:, :, :] += colored[:, :, 0:3]
        rgb_canvas_2d.clip(min=0, max=1, out=rgb_canvas_2d)

    def reconstruct_image_3d(self, time_point: TimePoint, rgb_canvas_3d: ndarray):
        """Draws the labels in color to the rgb image."""
        if self._segmented_channel is None:
            return  # Nothing to draw
        if self._display_settings.image_channel not in {self._channel_1, self._channel_2, self._segmented_channel}:
            return  # Nothing to draw for this channel
        if self._segmented_channel == self._display_settings.image_channel:
            # Avoid drawing on top of the same image
            rgb_canvas_3d[:, :, :, 0:3] = 0
            if rgb_canvas_3d.shape[-1] == 4:
                rgb_canvas_3d[:, :, :, 3] = 1  # Also erase alpha channel

        label_image = self._experiment.images.get_image_stack(self._time_point, self._segmented_channel)
        if label_image is None:
            return  # Nothing to show for this time point
        colored: ndarray = self._label_colormap(label_image.flatten())
        colored = colored.reshape((rgb_canvas_3d.shape[0], rgb_canvas_3d.shape[1], rgb_canvas_3d.shape[2], 4))
        rgb_canvas_3d[:, :, :, :] += colored[:, :, :, 0:3]
        rgb_canvas_3d.clip(min=0, max=1, out=rgb_canvas_3d)

    def _get_figure_title(self) -> str:
        return (f"Intensity measurement (pre-existing segmentation)\n"
                f"Time point {self._time_point.time_point_number()}    (z={self._get_figure_title_z_str()}, "
                f"c={self._display_settings.image_channel.index_one})")
