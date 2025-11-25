import json
import math
import os
from datetime import datetime
from typing import NamedTuple, Tuple, Optional, Iterable, Set, Callable, Sized, Dict

import keras
import numpy
import skimage
import tifffile
from skimage.feature import peak_local_max

import organoid_tracker.imaging.io
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Images, Image
from organoid_tracker.core.position import Position
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.neural_network.image_loading import fill_none_images_with_copies, extract_patch_array
from organoid_tracker.neural_network.position_detection_cnn.loss_functions import loss, position_precision, \
    position_recall, overcount
from organoid_tracker.neural_network.position_detection_cnn.peak_calling import reconstruct_volume


class _PredictionPatch(NamedTuple):
    """A crop of an image along with the coordinates it was taken from."""

    array: numpy.ndarray  # Raw image data, normalized between 0 and 1, but not yet resized to model input size
    corner_zyx: Tuple[int, int, int]  # Coordinates of the corner of the patch in the full image
    time_point: TimePoint  # Time point of the image

    # Scale factors between image and model input.
    scale_factors_zyx: Tuple[float, float, float]

    # Pixels this close to the border are buffer and should be ignored in predictions.
    # The buffer is in model pixels, so after resizing, and not the original image pixels.
    buffer_zyx_px: Tuple[int, int, int]

    # Size of the full image the patch was taken from
    full_image_size_zyx: Tuple[int, int, int]


class _Autosaver:
    """Used to autosave positions at regular intervals."""

    _output_file: Optional[str] = None
    _last_autosave_time: Optional[datetime] = None

    def set_output_file(self, filename: Optional[str]):
        """Sets the output file for saving. If filename is None, autosaving is disabled."""
        self._output_file = filename

    def autosave_after_interval(self, experiment: Experiment) -> bool:
        """Saves the positions to the output file if autosaving is enabled and enough time has passed since the
        last autosave.
        Returns True if an autosave was performed, False otherwise.
        """
        if self._output_file is None:
            return False
        if self._last_autosave_time is None or (datetime.now() - self._last_autosave_time).total_seconds() > 600:
            self._last_autosave_time = datetime.now()
            organoid_tracker.imaging.io.save_data_to_json(experiment, self._output_file)
            return True
        return False

    def save(self, experiment: Experiment):
        """Saves the positions to the output file if autosaving is enabled, regardless of when the last autosave was."""
        if self._output_file is None:
            return

        self._last_autosave_time = datetime.now()
        organoid_tracker.imaging.io.save_data_to_json(experiment, self._output_file)


class _DebugPredictions:
    """Class for storing and saving full prediction volumes for debugging purposes."""

    _full_predictions: Optional[numpy.ndarray] = None
    _output_file: Optional[str] = None

    def set_output_file(self, filename: Optional[str]):
        """Sets the output file for saving full predictions. If filename is None, debug storage is disabled."""
        self._output_file = filename

    def add_patch(self, prediction_patch: _PredictionPatch, predictions_array: numpy.ndarray):
        if self._output_file is None:
            return  # Debug storage not enabled

        if self._full_predictions is None:
            # Initialize full predictions storage
            self._full_predictions = numpy.zeros(prediction_patch.full_image_size_zyx, dtype=numpy.float32)

        # The predictions_array will have the shape the model expects, so we need to resize it back to the original image scale
        size_for_resizing = (
            int(predictions_array.shape[0] / prediction_patch.scale_factors_zyx[0]),
            int(predictions_array.shape[1] / prediction_patch.scale_factors_zyx[1]),
            int(predictions_array.shape[2] / prediction_patch.scale_factors_zyx[2]),
        )
        resized_predictions = skimage.transform.resize(predictions_array, output_shape=size_for_resizing, order=0,
                                                       clip=False, preserve_range=True, anti_aliasing=False)

        # We cut off the buffer area on the lower coords
        # (at the higher coords we just rely on overwriting by later patches)
        buffer_size_resized_zyx = (
            int(prediction_patch.buffer_zyx_px[0] / prediction_patch.scale_factors_zyx[0]),
            int(prediction_patch.buffer_zyx_px[1] / prediction_patch.scale_factors_zyx[1]),
            int(prediction_patch.buffer_zyx_px[2] / prediction_patch.scale_factors_zyx[2]),
        )
        target_zyx = (prediction_patch.corner_zyx[0] + buffer_size_resized_zyx[0],
                      prediction_patch.corner_zyx[1] + buffer_size_resized_zyx[1],
                      prediction_patch.corner_zyx[2] + buffer_size_resized_zyx[2])
        resized_predictions = resized_predictions[buffer_size_resized_zyx[0]:,
        buffer_size_resized_zyx[1]:,
        buffer_size_resized_zyx[2]:]

        # Calculate how much of the resized predictions fit into the full predictions
        # (patches have a minimum size, so at the edges of the full image they may be too large)
        size_z = min(self._full_predictions.shape[0] - target_zyx[0], resized_predictions.shape[0])
        size_y = min(self._full_predictions.shape[1] - target_zyx[1], resized_predictions.shape[1])
        size_x = min(self._full_predictions.shape[2] - target_zyx[2], resized_predictions.shape[2])
        resized_predictions = resized_predictions[0:size_z, 0:size_y, 0:size_x]

        # Store into full predictions
        self._full_predictions[target_zyx[0]:target_zyx[0] + size_z,
        target_zyx[1]:target_zyx[1] + size_y,
        target_zyx[2]:target_zyx[2] + size_x] = resized_predictions

    def save_full_predictions(self):
        """Saves the full predictions to a .tif file. Does nothing if debug storage is not enabled."""
        if self._full_predictions is None or self._output_file is None:
            return  # Debug storage not enabled
        tifffile.imwrite(self._output_file, self._full_predictions, compression=tifffile.COMPRESSION.ADOBE_DEFLATE,
                         compressionargs={"level": 9})


def _split_into_patches(time_point: TimePoint, full_images: Dict[TimePoint, Image], *,
                        patch_shape_zyx_px: Tuple[int, int, int],
                        buffer_size_zyx_px: Tuple[int, int, int],
                        scale_factors_zyx: Tuple[float, float, float],
                        intensity_quantiles: Tuple[float, float]) -> Iterable[_PredictionPatch]:
    """patch_shape_z needs to match what the model expect, and patch_shape_y and x should be a multiple of 32."""

    # Create a dictionary of all full images in the time window
    time_point_image: Image = full_images.get(time_point)
    if time_point_image is None:
        return  # No image at the center time point
    fill_none_images_with_copies(
        full_images)  # If images are missing (start or end of movie), fill with nearest available image

    min_intensity = float(
        numpy.min([numpy.quantile(image.array, intensity_quantiles[0]) for image in full_images.values()]))
    max_intensity = float(
        numpy.max([numpy.quantile(image.array, intensity_quantiles[1]) for image in full_images.values()]))

    # Calculate patch shape and buffer size in the input image pixels (instead of the pixels the model expects)
    patch_shape_zyx_image_px = (int(patch_shape_zyx_px[0] / scale_factors_zyx[0]),
                                int(patch_shape_zyx_px[1] / scale_factors_zyx[1]),
                                int(patch_shape_zyx_px[2] / scale_factors_zyx[2]))
    buffer_size_zyx_image_px = (int(buffer_size_zyx_px[0] / scale_factors_zyx[0]),
                                int(buffer_size_zyx_px[1] / scale_factors_zyx[1]),
                                int(buffer_size_zyx_px[2] / scale_factors_zyx[2]))
    patch_shape_without_buffer_zyx_image_px = [patch_shape_zyx_image_px[0] - 2 * buffer_size_zyx_image_px[0],
                                               patch_shape_zyx_image_px[1] - 2 * buffer_size_zyx_image_px[1],
                                               patch_shape_zyx_image_px[2] - 2 * buffer_size_zyx_image_px[2]]

    # Make the patches
    for z_start in range(time_point_image.min_z - buffer_size_zyx_image_px[0],
                         time_point_image.limit_z + buffer_size_zyx_image_px[0],
                         patch_shape_without_buffer_zyx_image_px[0]):
        for y_start in range(time_point_image.min_y - buffer_size_zyx_image_px[1],
                             time_point_image.limit_y + buffer_size_zyx_image_px[1],
                             patch_shape_without_buffer_zyx_image_px[1]):
            for x_start in range(time_point_image.min_x - buffer_size_zyx_image_px[2],
                                 time_point_image.limit_x + buffer_size_zyx_image_px[2],
                                 patch_shape_without_buffer_zyx_image_px[2]):
                array = extract_patch_array(full_images, (z_start, y_start, x_start), patch_shape_zyx_image_px)

                # Normalize patch
                array /= (max_intensity - min_intensity)
                array -= min_intensity / (max_intensity - min_intensity)
                numpy.clip(array, 0.0, 1.0, out=array)

                yield _PredictionPatch(array=array,
                                       corner_zyx=(z_start, y_start, x_start),
                                       time_point=time_point,
                                       scale_factors_zyx=scale_factors_zyx,
                                       buffer_zyx_px=buffer_size_zyx_px,
                                       full_image_size_zyx=time_point_image.array.shape)


def _count_patches(time_point: TimePoint, full_images: Dict[TimePoint, Image], *,
                   patch_shape_zyx_px: Tuple[int, int, int],
                   buffer_size_zyx_px: Tuple[int, int, int],
                   scale_factors_zyx: Tuple[float, float, float]) -> int:
    """Counts how many patches will be generated by _split_into_patches()."""
    time_point_image = full_images.get(time_point)

    # Calculate patch shape and buffer size in the input image pixels (instead of the pixels the model expects)
    patch_shape_zyx_image_px = (int(patch_shape_zyx_px[0] / scale_factors_zyx[0]),
                                int(patch_shape_zyx_px[1] / scale_factors_zyx[1]),
                                int(patch_shape_zyx_px[2] / scale_factors_zyx[2]))
    buffer_size_zyx_image_px = (int(buffer_size_zyx_px[0] / scale_factors_zyx[0]),
                                int(buffer_size_zyx_px[1] / scale_factors_zyx[1]),
                                int(buffer_size_zyx_px[2] / scale_factors_zyx[2]))
    patch_shape_without_buffer_zyx_image_px = [patch_shape_zyx_image_px[0] - 2 * buffer_size_zyx_image_px[0],
                                               patch_shape_zyx_image_px[1] - 2 * buffer_size_zyx_image_px[1],
                                               patch_shape_zyx_image_px[2] - 2 * buffer_size_zyx_image_px[2]]

    # Count patches
    z_size = time_point_image.limit_z + buffer_size_zyx_image_px[0] - (time_point_image.min_z - buffer_size_zyx_image_px[0])
    y_size = time_point_image.limit_y + buffer_size_zyx_image_px[1] - (time_point_image.min_y - buffer_size_zyx_image_px[1])
    x_size = time_point_image.limit_x + buffer_size_zyx_image_px[2] - (time_point_image.min_x - buffer_size_zyx_image_px[2])
    return (math.ceil(z_size / patch_shape_without_buffer_zyx_image_px[0]) *
            math.ceil(y_size / patch_shape_without_buffer_zyx_image_px[1]) *
            math.ceil(x_size / patch_shape_without_buffer_zyx_image_px[2]))


class PositionModel(NamedTuple):
    """A position prediction model loaded from disk.

    You can use it to make predictions using the predict function. You can load such a model from disk using the
    load_position_model function below.
    """

    keras_model: keras.Model
    time_window: Tuple[int, int]

    def predict_positions(self, experiment: Experiment, *,
                          debug_folder_experiment: Optional[str] = None,
                          image_channels: Optional[Set[ImageChannel]] = None,
                          peak_min_distance_px: int = 6,
                          patch_shape_unbuffered_yx: Tuple[int, int] = (320, 320),
                          buffer_size_zyx: Tuple[int, int, int] = (1, 32, 32),
                          threshold: float = 0.1,
                          mid_layers: int = 5,
                          scale_factors_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                          intensity_quantiles: Tuple[float, float] = (0.01, 0.99),
                          time_points: Optional[Iterable[TimePoint] | Sized] = None,
                          progress_callback: Callable[[float], None] = lambda _: None,
                          output_file: Optional[str] = None):
        """Predict positions for the given experiment.

        Args:
            experiment: The experiment to predict positions for.
            debug_folder_experiment: If given, a folder to store the raw peaks.
            image_channels: If given, the image channels to use for prediction. If None, the first channel is used.
            If multiple channels are given, they are summed together.
            peak_min_distance_px: Minimum distance in pixels between detected positions.
            patch_shape_unbuffered_yx: The shape of the patches to use for prediction in y and x. Must be a multiple of 32.
            buffer_size_zyx: The buffer size to use when splitting images into patches. Positions detected within the buffer
                area are ignored. Added on top of patch_shape_unbuffered_yx.
            threshold: Threshold for peak finding.
            automatically calculate it from the model resolution, such that the resolution becomes isotropic.
            mid_layers: Interpolate this many layers between the original layers for peak detection.
            scale_factors_zyx: Scale factors to apply to the input images to reach the target resolution of the model.
            intensity_quantiles: The quantiles to use for intensity normalization. Normalization is done per time point
            on the full 3D image before splitting into patches.
            time_points: If given, only predict positions for these time points. If None, predict for all time points.
            progress_callback: A callback function that is called with the progress (between 0.0 and 1.0).
            output_file: If given, the file to save the predicted positions to. Otherwise, positions are not saved to
            disk, and just set in the experiment.
        """

        # Check if images were loaded
        if not experiment.images.image_loader().has_images():
            raise ValueError(
                f"No images were found for experiment \"{experiment.name}\". Please check the configuration file and make"
                f" sure that you have stored images at the specified location.")

        # Set up debug folder
        if debug_folder_experiment is not None:
            os.makedirs(debug_folder_experiment, exist_ok=True)

        # Set up autosaving
        autosaver = _Autosaver()
        autosaver.set_output_file(output_file)
        is_autosaved = False

        # Edit image channels if necessary
        if image_channels is None:
            image_channels = {ImageChannel(index_one=1)}

        # Create an image loader where the first channel is the one we want to use for predictions
        image_loader = experiment.images.image_loader()
        if image_channels != [ImageChannel(index_one=1)]:
            image_loader = ChannelSummingImageLoader(experiment.images.image_loader(), [image_channels])
        images = Images()
        images.image_loader(image_loader)
        images.offsets = experiment.images.offsets
        images.set_resolution(experiment.images.resolution())

        patch_shape_z = self.keras_model.layers[0].batch_shape[1]
        patch_shape_y = patch_shape_unbuffered_yx[0] + buffer_size_zyx[1] * 2
        patch_shape_x = patch_shape_unbuffered_yx[1] + buffer_size_zyx[2] * 2
        patch_shape_zyx = (patch_shape_z, patch_shape_y, patch_shape_x)

        # Loop over all time points
        if time_points is None:
            time_points = experiment.images.time_points()
        time_points_count = len(time_points)
        time_points_done = 0
        for time_point in time_points:
            if experiment.positions.count_positions(time_point=time_point) > 0:
                continue  # Skip time points that already have positions
            print(time_point.time_point_number(), end="  ", flush=True)
            debug_predictions = _DebugPredictions()
            if debug_folder_experiment is not None:
                debug_predictions.set_output_file(
                    os.path.join(debug_folder_experiment, f"image_{time_point.time_point_number()}.tif"))

            full_images = dict()
            for dt in range(self.time_window[0], self.time_window[1] + 1):
                time_point_dt = TimePoint(time_point.time_point_number() + dt)
                full_images[time_point_dt] = images.get_image(time_point_dt, ImageChannel(index_one=1))
            patch_count = _count_patches(time_point, full_images,
                                         patch_shape_zyx_px=patch_shape_zyx,
                                         buffer_size_zyx_px=buffer_size_zyx,
                                         scale_factors_zyx=scale_factors_zyx)
            patches_done = 0
            for patch in _split_into_patches(time_point, full_images,
                                             patch_shape_zyx_px=patch_shape_zyx,
                                             buffer_size_zyx_px=buffer_size_zyx,
                                             scale_factors_zyx=scale_factors_zyx,
                                             intensity_quantiles=intensity_quantiles):
                # Resize image using scipy zoom
                patch_time_point_count = patch.array.shape[-1]
                input_array = numpy.zeros((*patch_shape_zyx, patch_time_point_count))
                for i in range(patch_time_point_count):
                    input_array[:, :, :, i] = skimage.transform.resize(patch.array[:, :, :, i],
                                                                       output_shape=patch_shape_zyx, order=0,
                                                                       clip=False, preserve_range=True,
                                                                       anti_aliasing=False)

                # Call the model
                prediction = \
                keras.ops.convert_to_numpy(self.keras_model(input_array[numpy.newaxis, ...], training=False))[0, :, :, :, 0]

                debug_predictions.add_patch(patch, prediction)

                # Interpolate between layers for peak detection
                prediction, z_divisor = reconstruct_volume(prediction, mid_layers)
                coordinates = peak_local_max(prediction, min_distance=peak_min_distance_px,
                                             threshold_abs=threshold, exclude_border=False)
                for coordinate in coordinates:
                    # Back to coords of the prediction input
                    prediction_z = coordinate[0] / z_divisor - 1
                    prediction_y = coordinate[1]
                    prediction_x = coordinate[2]

                    # Check if inside buffer area
                    if (prediction_z < patch.buffer_zyx_px[0] or
                            prediction_z >= prediction.shape[0] - patch.buffer_zyx_px[0] or
                            prediction_y < patch.buffer_zyx_px[1] or
                            prediction_y >= prediction.shape[1] - patch.buffer_zyx_px[1] or
                            prediction_x < patch.buffer_zyx_px[2] or
                            prediction_x >= prediction.shape[2] - patch.buffer_zyx_px[2]):
                        continue  # Inside buffer, ignore

                    # Back to coords of the full image
                    full_image_z = int(prediction_z / patch.scale_factors_zyx[0] + patch.corner_zyx[0])
                    full_image_y = int(prediction_y / patch.scale_factors_zyx[1] + patch.corner_zyx[1])
                    full_image_x = int(prediction_x / patch.scale_factors_zyx[2] + patch.corner_zyx[2])

                    # Bounds check for full image (the last patches may go beyond the image size, because patches have a minimum size)
                    full_image_size_zyx = patch.full_image_size_zyx
                    if full_image_z >= full_image_size_zyx[0] or full_image_y >= full_image_size_zyx[
                        1] or full_image_x >= full_image_size_zyx[2]:
                        continue

                    experiment.positions.add(
                        Position(full_image_x, full_image_y, full_image_z, time_point=patch.time_point))

                # Report progress
                patches_done += 1
                total_iterations = patch_count * time_points_count
                iterations_done = time_points_done * patch_count + patches_done
                progress_callback(iterations_done / total_iterations)
            time_points_done += 1
            debug_predictions.save_full_predictions()
            is_autosaved = autosaver.autosave_after_interval(experiment)
        if not is_autosaved:
            autosaver.save(experiment)
        progress_callback(1.0)


def load_position_model(model_folder: str) -> PositionModel:
    """Load a position prediction model from the given folder."""
    model_folder = os.path.abspath(model_folder)

    keras_model: keras.Model = keras.saving.load_model(os.path.join(model_folder, "model.keras"),
                                                       custom_objects={"loss": loss,
                                                                       "position_precision": position_precision,
                                                                       "position_recall": position_recall,
                                                                       "overcount": overcount})

    # Set relevant parameters
    if not os.path.isfile(os.path.join(model_folder, "settings.json")):
        raise ValueError("Error: no settings.json found in model folder.")
    with open(os.path.join(model_folder, "settings.json")) as file_handle:
        json_contents = json.load(file_handle)
        if json_contents["type"] != "positions":
            raise ValueError("Error: model is made for working with " + str(json_contents["type"]) + ", not positions")
        time_window = json_contents["time_window"]

    # Convert time_window to tuple of ints
    time_window = (int(time_window[0]), int(time_window[1]))

    return PositionModel(keras_model=keras_model, time_window=time_window)
