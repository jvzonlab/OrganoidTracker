from datetime import datetime
import json
import os
from typing import NamedTuple, Tuple, Optional, Iterable, Set, Dict

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
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import cropper
from organoid_tracker.neural_network.position_detection_cnn.loss_functions import loss, position_precision, \
    position_recall, overcount
from organoid_tracker.neural_network.position_detection_cnn.peak_calling import reconstruct_volume

# Default target resolution for position models
# New models should always specify their target resolution in settings.json
_DEFAULT_TARGET_RESOLUTION_ZYX_UM = (2.0, 0.32, 0.32)


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

    def set_full_storage_size(self, shape: Optional[Tuple[int, int, int]]):
        """Sets up storage for full predictions. If shape is None, debug storage is disabled.
        Throws ValueError if output file hasn't been set before, but a shape is given.
        """
        if shape is None:
            self._full_predictions = None
            return
        if self._output_file is None:
            raise ValueError("Output file must be set before setting full storage size.")
        self._full_predictions = numpy.zeros(shape, dtype=numpy.float32)

    def set_output_file(self, filename: Optional[str]):
        """Sets the output file for saving full predictions. If filename is None, debug storage is disabled."""
        self._output_file = filename

    def add_patch(self, prediction_patch: _PredictionPatch, predictions_array: numpy.ndarray):
        if self._full_predictions is None:
            return  # Debug storage not enabled

        # The predictions_array will have the shape the model expects, so we need to resize it back to the original image scale
        size_for_resizing = (
            int(predictions_array.shape[0] / prediction_patch.scale_factors_zyx[0]),
            int(predictions_array.shape[1] / prediction_patch.scale_factors_zyx[1]),
            int(predictions_array.shape[2] / prediction_patch.scale_factors_zyx[2]),
        )
        resized_predictions = skimage.transform.resize(predictions_array, output_shape=size_for_resizing, order=0, clip=False, preserve_range=True, anti_aliasing=False)

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
        self._full_predictions[target_zyx[0]:target_zyx[0]+size_z,
                               target_zyx[1]:target_zyx[1]+size_y,
                               target_zyx[2]:target_zyx[2]+size_x] = resized_predictions

    def save_full_predictions(self):
        """Saves the full predictions to a .tif file. Does nothing if debug storage is not enabled."""
        if self._full_predictions is None or self._output_file is None:
            return  # Debug storage not enabled
        tifffile.imwrite(self._output_file, self._full_predictions, compression=tifffile.COMPRESSION.ADOBE_DEFLATE,
                         compressionargs={"level": 9})


def _fill_none_images_with_copies(full_images: Dict[TimePoint, Image]):
    """At the start or end of a movie, some time points may be missing. Fill these with copies of the nearest available
    image."""
    time_points = list(full_images.keys())
    for time_point in time_points:
        if full_images[time_point] is not None:
            continue
        # Find nearest available image
        offset = 1
        while True:
            time_point_before = TimePoint(time_point.time_point_number() - offset)
            image_before = full_images.get(time_point_before)
            if image_before is not None:
                full_images[time_point] = image_before
                break

            time_point_after = TimePoint(time_point.time_point_number() + offset)
            image_after = full_images.get(time_point_after)
            if image_after is not None:
                full_images[time_point] = image_after
                break

            offset += 1
            if offset > len(time_points):
                raise ValueError("No images available to fill missing time points.")


def _split_into_patches(images: Images, time_point: TimePoint,
                        time_window: Tuple[int, int],
                        patch_shape_zyx_px: Tuple[int, int, int],
                        buffer_size_zyx_px: Tuple[int, int, int],
                        target_resolution_zyx_um: Tuple[float, float, float]) -> Iterable[_PredictionPatch]:
    """patch_shape_z needs to match what the model expect, and patch_shape_y and x should be a multiple of 32."""

    # Create a dictionary of all full images in the time window
    full_images = dict()
    for dt in range(time_window[0], time_window[1] + 1):
        time_point_dt = TimePoint(time_point.time_point_number() + dt)
        full_images[time_point_dt] = images.get_image(time_point_dt, ImageChannel(index_one=1))
    time_point_image = full_images.get(time_point)
    if time_point_image is None:
        return  # No image at the center time point
    _fill_none_images_with_copies(full_images)  # If images are missing (start or end of movie), fill with nearest available image

    min_intensity = float(numpy.min([numpy.quantile(image.array, 0.01) for image in full_images.values()]))
    max_intensity = float(numpy.max([numpy.quantile(image.array, 0.99) for image in full_images.values()]))

    # Calculate scaling for the model's target resolution
    image_resolution = images.resolution()
    scale_factors_zyx = (
        image_resolution.pixel_size_z_um / target_resolution_zyx_um[0],
        image_resolution.pixel_size_y_um / target_resolution_zyx_um[1],
        image_resolution.pixel_size_x_um / target_resolution_zyx_um[2]
    )
    # So if the image has a resolution of 0.16 um/px in x, and the model has a target resolution of 0.32 um/px, the scale factor is 0.5.
    # If a patch is 384 px wide in x, we need to take 384 / 0.5 = 768 px wide from the original image.

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
    for z_start in range(time_point_image.min_z - buffer_size_zyx_image_px[0], time_point_image.limit_z + buffer_size_zyx_image_px[0], patch_shape_without_buffer_zyx_image_px[0]):
        for y_start in range(time_point_image.min_y - buffer_size_zyx_image_px[1], time_point_image.limit_y + buffer_size_zyx_image_px[1], patch_shape_without_buffer_zyx_image_px[1]):
            for x_start in range(time_point_image.min_x - buffer_size_zyx_image_px[2], time_point_image.limit_x + buffer_size_zyx_image_px[2], patch_shape_without_buffer_zyx_image_px[2]):
                array = _extract_patch_array(full_images, (z_start, y_start, x_start), patch_shape_zyx_image_px)

                # Normalize patch
                array /= (max_intensity - min_intensity)
                array -= min_intensity / (max_intensity - min_intensity)
                numpy.clip(array, 0.0, 1.0, out=array)

                yield _PredictionPatch(array=array,
                                       corner_zyx=(z_start, y_start, x_start),
                                       time_point=time_point,
                                       scale_factors_zyx=scale_factors_zyx,
                                       buffer_zyx_px=buffer_size_zyx_px)


def _extract_patch_array(full_images: Dict[TimePoint, Image], start_zyx: Tuple[int, int, int],
                   patch_shape_zyx: Tuple[int, int, int]) -> numpy.ndarray:
    """Extracts a patch from the full images. Coordinates are assumed to be in position coordinates. Offsets of the
    images are taken into account. full_images must contain a continuous set of time points.
    """

    min_time_point = min(full_images.keys())
    output_array = numpy.zeros((patch_shape_zyx[0], patch_shape_zyx[1], patch_shape_zyx[2], len(full_images)), dtype=numpy.float32)

    for time_point_dt, image in full_images.items():
        dt_index = time_point_dt.time_point_number() - min_time_point.time_point_number()
        offset = image.offset
        x_start = int(start_zyx[2] - offset.x)
        y_start = int(start_zyx[1] - offset.y)
        z_start = int(start_zyx[0] - offset.z)

        cropper.crop_3d(image.array, x_start, y_start, z_start, output_array[:, :, :, dt_index])

    return output_array


class PositionModel(NamedTuple):
    """A position prediction model loaded from disk.

    You can use it to make predictions using the predict function. You can load such a model from disk using the
    load_position_model function below.
    """

    keras_model: keras.Model
    time_window: Tuple[int, int]
    target_resolution_zyx_um: Tuple[float, float, float]

    def predict(self, experiment: Experiment, *,
                debug_folder_experiment: Optional[str] = None,
                image_channels: Optional[Set[ImageChannel]] = None,
                peak_min_distance_px: int = 6,
                patch_shape_unbuffered_yx: Tuple[int, int] = (320, 320),
                buffer_size_zyx: Tuple[int, int, int] = (1, 32, 32),
                threshold: float = 0.1,
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
            output_file: If given, the file to save the predicted positions to. Otherwise, positions are not saved to
            disk, and just set in the experiment.
        """

        # Auto-calculate mid_layers to get isotropic resolution
        mid_layers = max(int(self.target_resolution_zyx_um[0] / self.target_resolution_zyx_um[1] - 1), 0)

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
        for time_point in experiment.images.time_points():
            if experiment.positions.count_positions(time_point=time_point) > 0:
                continue  # Skip time points that already have positions
            print(time_point.time_point_number(), end="  ", flush=True)
            debug_predictions = _DebugPredictions()
            if debug_folder_experiment is not None:
                debug_predictions.set_output_file(os.path.join(debug_folder_experiment, f"image_{time_point.time_point_number()}.tif"))
                debug_predictions.set_full_storage_size(images.image_loader().get_image_size_zyx())

            for patch in _split_into_patches(images, time_point, self.time_window, patch_shape_zyx, buffer_size_zyx, self.target_resolution_zyx_um):
                # Resize image using scipy zoom
                time_point_count = patch.array.shape[-1]
                input_array = numpy.zeros((*patch_shape_zyx, time_point_count))
                for i in range(time_point_count):
                    input_array[:, :, :, i] = skimage.transform.resize(patch.array[:, :, :, i], output_shape=patch_shape_zyx, order=0, clip=False, preserve_range=True, anti_aliasing=False)

                # Call the model
                prediction = keras.ops.convert_to_numpy(self.keras_model(input_array[numpy.newaxis, ...], training=False))[0, :, :, :, 0]

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

                    experiment.positions.add(Position(full_image_x, full_image_y, full_image_z, time_point=patch.time_point))
            debug_predictions.save_full_predictions()
            is_autosaved = autosaver.autosave_after_interval(experiment)
        if not is_autosaved:
            autosaver.save(experiment)


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

        if "target_resolution_zyx_um" in json_contents:
            target_resolution_zyx_um = tuple(json_contents["target_resolution_zyx_um"])
        else:
            target_resolution_zyx_um = _DEFAULT_TARGET_RESOLUTION_ZYX_UM

    # Convert time_window to tuple of ints
    time_window = (int(time_window[0]), int(time_window[1]))

    return PositionModel(keras_model=keras_model, time_window=time_window, target_resolution_zyx_um=target_resolution_zyx_um)

