import json
import os
from typing import Tuple, NamedTuple, Iterable, List

import keras
import numpy
import skimage

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image, Images
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.neural_network import DEFAULT_TARGET_RESOLUTION_ZYX_UM
from organoid_tracker.neural_network.image_loading import fill_none_images_with_copies, extract_patch_array


class _PredictionPatch(NamedTuple):
    position: Position
    array: numpy.ndarray


def _split_into_patches(images: Images, time_point: TimePoint, positions: Iterable[Position],
                        time_window: Tuple[int, int],
                        patch_shape_zyx_px: Tuple[int, int, int],
                        target_resolution_zyx_um: Tuple[float, float, float]) -> Iterable[_PredictionPatch]:
    """patch_shape_z needs to match what the model expect, and patch_shape_y and x should be a multiple of 32."""

    # Create a dictionary of all full images in the time window
    full_images = dict()
    for dt in range(time_window[0], time_window[1] + 1):
        time_point_dt = TimePoint(time_point.time_point_number() + dt)
        full_images[time_point_dt] = images.get_image(time_point_dt, ImageChannel(index_one=1))
    time_point_image: Image = full_images.get(time_point)
    if time_point_image is None:
        return  # No image at the center time point
    fill_none_images_with_copies(full_images)  # If images are missing (start or end of movie), fill with nearest available image

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
    # If a patch is 64 px wide in x, we need to take 64 / 0.5 = 128 px wide from the original image.

    # Calculate patch shape in the input image pixels (instead of the pixels the model expects)
    patch_shape_zyx_image_px = (int(patch_shape_zyx_px[0] / scale_factors_zyx[0]),
                                int(patch_shape_zyx_px[1] / scale_factors_zyx[1]),
                                int(patch_shape_zyx_px[2] / scale_factors_zyx[2]))
    # Make the patches
    for position in positions:
        z_start = int(round(position.z - patch_shape_zyx_image_px[0] / 2))
        y_start = int(round(position.y - patch_shape_zyx_image_px[1] / 2))
        x_start = int(round(position.x - patch_shape_zyx_image_px[2] / 2))

        array = extract_patch_array(full_images, (z_start, y_start, x_start), patch_shape_zyx_image_px)

        # Normalize patch
        array /= (max_intensity - min_intensity)
        array -= min_intensity / (max_intensity - min_intensity)
        numpy.clip(array, 0.0, 1.0, out=array)

        yield _PredictionPatch(array=array, position=position)


class DivisionModel(NamedTuple):
    keras_model: keras.Model
    time_window: Tuple[int, int]
    patch_shape_zyx: Tuple[int, int, int]
    platt_scaling: float
    platt_intercept: float
    target_resolution_zyx_um: Tuple[float, float, float]

    def _iterate_patches(self, images: Images, positions: PositionCollection) -> Iterable[_PredictionPatch]:
        for time_point in images.time_points():
            print(time_point.time_point_number(), end="  ", flush=True)
            positions_of_time_point = positions.of_time_point(time_point)
            if len(positions_of_time_point) == 0:
                continue
            yield from _split_into_patches(images, time_point, positions_of_time_point,
                                               self.time_window, self.patch_shape_zyx,
                                               self.target_resolution_zyx_um)

    def predict_divisions(self, experiment: Experiment, *,
                          batch_size: int = 32,
                          image_channels: List[ImageChannel] = None):

        # Check if images were loaded
        if not experiment.images.image_loader().has_images():
            raise ValueError(
                f"No images were found for experiment \"{experiment.name}\". Please check the configuration file and make"
                f" sure that you have stored images at the specified location.")

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

        # Do predictions
        patch_list: List[_PredictionPatch] = list()
        for patch in self._iterate_patches(images, experiment.positions):
            patch_list.append(patch)
            if len(patch_list) == batch_size:
                self._predict_batch(experiment, patch_list)
                patch_list.clear()

        if len(patch_list) > 0:
            # Predict any remaining patches
            self._predict_batch(experiment, patch_list)


    def _predict_batch(self, experiment: Experiment, patch_list: List[_PredictionPatch]):
        # Create input array for model
        time_window_size = self.time_window[1] - self.time_window[0] + 1
        input_array = numpy.zeros((len(patch_list),
                                   self.patch_shape_zyx[0],
                                   self.patch_shape_zyx[1],
                                   self.patch_shape_zyx[2],
                                   time_window_size), dtype=numpy.float32)
        for i, patch in enumerate(patch_list):
            for j in range(time_window_size):
                input_array[i, :, :, :, j] = skimage.transform.resize(patch.array[:, :, :, j], self.patch_shape_zyx, order=0,
                                                                      clip=False, preserve_range=True, anti_aliasing=False)

        # Predict
        raw_predictions = keras.ops.convert_to_numpy(self.keras_model(input_array, training=False))
        raw_predictions = raw_predictions.flatten()

        # Apply Platt scaling
        eps = 10 ** -10
        likelihoods = self.platt_intercept + self.platt_scaling * (
                    numpy.log10(raw_predictions + eps) - numpy.log10(1 - raw_predictions + eps))
        scaled_predictions = (10 ** likelihoods) / (1 + 10 ** likelihoods)

        # Store predictions
        for i, patch in enumerate(patch_list):
            experiment.positions.set_position_data(patch.position, "division_probability", float(scaled_predictions[i]))
            experiment.positions.set_position_data(patch.position, "division_penalty", float(-likelihoods[i]))


def load_division_model(model_folder: str) -> DivisionModel:
    """Load a position prediction model from the given folder."""
    model_folder = os.path.abspath(model_folder)

    keras_model: keras.Model = keras.saving.load_model(os.path.join(model_folder, "model.keras"))

    # Set relevant parameters
    if not os.path.isfile(os.path.join(model_folder, "settings.json")):
        raise ValueError("Error: no settings.json found in model folder.")
    with open(os.path.join(model_folder, "settings.json")) as file_handle:
        json_contents = json.load(file_handle)
        if json_contents["type"] != "divisions":
            print("Error: model at " + model_folder + " is made for working with " + str(
                json_contents["type"]) + ", not divisions")
            exit(1)
        time_window = json_contents["time_window"]
        patch_shape_zyx = json_contents["patch_shape_zyx"]

        scaling = json_contents["platt_scaling"] if "platt_scaling" in json_contents else 1
        intercept = json_contents["platt_intercept"] if "platt_intercept" in json_contents else 0
        intercept = numpy.log10(numpy.exp(intercept))

        if "target_resolution_zyx_um" in json_contents:
            target_resolution_zyx_um = tuple(json_contents["target_resolution_zyx_um"])
        else:
            target_resolution_zyx_um = DEFAULT_TARGET_RESOLUTION_ZYX_UM

    # Convert to tuples
    time_window = (int(time_window[0]), int(time_window[1]))
    patch_shape_zyx = (int(patch_shape_zyx[0]), int(patch_shape_zyx[1]), int(patch_shape_zyx[2]))

    return DivisionModel(keras_model=keras_model,
                         time_window=time_window,
                         target_resolution_zyx_um=target_resolution_zyx_um,
                         patch_shape_zyx=patch_shape_zyx,
                         platt_scaling=scaling,
                         platt_intercept=intercept)
