import json
import os
from typing import NamedTuple, Tuple, Set, List, Iterable, Dict

import keras
import numpy
import skimage

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Images, Image
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.linking import nearest_neighbor_linker
from organoid_tracker.neural_network.image_loading import fill_none_images_with_copies, extract_patch_array
from organoid_tracker.neural_network.link_detection_cnn.training_dataset import add_3d_coord


class _PredictionPatch(NamedTuple):
    array_a: numpy.ndarray  # First time point
    array_b: numpy.ndarray  # Second time point
    position_a: Position
    position_b: Position
    distance_zyx_px: Tuple[float, float, float]


class LinkModel(NamedTuple):
    keras_model: keras.Model
    time_window: Tuple[int, int]
    patch_shape_zyx: Tuple[int, int, int]
    platt_scaling: float
    platt_intercept: float

    def predict_links(self, experiment: Experiment, *,
                          batch_size: int = 32,
                          image_channels: Set[ImageChannel] = None,
                          scale_factors_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                          intensity_quantiles: Tuple[float, float] = (0.01, 0.99)):
        """Predict division probabilities for all links in the given experiment."""

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

        # Find possible links
        possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
        experiment.links = possible_links

        # Do predictions
        patch_list: List[_PredictionPatch] = list()
        for patch in self._iterate_patches(images, experiment.positions, possible_links, scale_factors_zyx=scale_factors_zyx,
                                           intensity_quantiles=intensity_quantiles):
            patch_list.append(patch)
            if len(patch_list) == batch_size:
                self._predict_batch(experiment, patch_list)
                patch_list.clear()

        if len(patch_list) > 0:
            # Predict any remaining patches
            self._predict_batch(experiment, patch_list)

    def _iterate_patches(self, images: Images, positions: PositionCollection,
                         possible_links: Links,
                         *,
                         scale_factors_zyx: Tuple[float, float, float],
                         intensity_quantiles: Tuple[float, float]) -> Iterable[_PredictionPatch]:

        experiment_nearest_neighbor = Experiment()
        experiment_nearest_neighbor.images = images
        experiment_nearest_neighbor.positions = positions

        for time_point in images.time_points():
            if images.get_image_stack(time_point + 1) is None:
                break  # At the end of the movie, cannot link to next time point

            print(time_point.time_point_number(), end="  ", flush=True)

            # Collect positions at this time point
            links_of_time_point = list()
            for position_of_time_point in positions.of_time_point(time_point):
                for future_position in possible_links.find_futures(position_of_time_point):
                    links_of_time_point.append((position_of_time_point, future_position))

            if len(links_of_time_point) == 0:
                continue

            yield from _split_into_patches(images, time_point, links_of_time_point, self.time_window,
                                           patch_shape_zyx_px=self.patch_shape_zyx,
                                           scale_factors_zyx=scale_factors_zyx,
                                           intensity_quantiles=intensity_quantiles)


    def _predict_batch(self, experiment: Experiment, patch_list: List[_PredictionPatch]):
        # Create input array for model (channels: time points + 3 CoordConv channels)
        time_window_size = self.time_window[1] - self.time_window[0] + 1
        input_array_a = keras.ops.zeros((len(patch_list),
                                    self.patch_shape_zyx[0],
                                    self.patch_shape_zyx[1],
                                    self.patch_shape_zyx[2],
                                    time_window_size + 3), dtype="float32")
        input_array_b = keras.ops.zeros_like(input_array_a)
        distances_zyx = keras.ops.zeros((len(patch_list), 3), dtype="float32")
        for i, patch in enumerate(patch_list):
            distance_zyx = keras.ops.convert_to_tensor(patch.distance_zyx_px, dtype="float32")

            # Resize patches to model input size
            patch_array_a = keras.ops.zeros((self.patch_shape_zyx[0], self.patch_shape_zyx[1], self.patch_shape_zyx[2], time_window_size), dtype="float32")
            patch_array_b = keras.ops.zeros((self.patch_shape_zyx[0], self.patch_shape_zyx[1], self.patch_shape_zyx[2], time_window_size), dtype="float32")
            for j in range(time_window_size):
                patch_array_a[:, :, :, j] = keras.ops.convert_to_tensor(
                    skimage.transform.resize(patch.array_a[:, :, :, j], self.patch_shape_zyx,
                                                                     order=0, clip=False, preserve_range=True,
                                                                     anti_aliasing=False))
                patch_array_b[:, :, :, j] = keras.ops.convert_to_tensor(
                    skimage.transform.resize(patch.array_b[:, :, :, j], self.patch_shape_zyx,
                                                                     order=0, clip=False, preserve_range=True,
                                                                     anti_aliasing=False))

            # Add CoordConv transform
            patch_array_a, patch_array_b = add_3d_coord(patch_array_a, patch_array_b, distance_zyx)

            # Store in batch array
            input_array_a[i] = patch_array_a
            input_array_b[i] = patch_array_b
            distances_zyx[i] = distance_zyx

        # Switch to GPU
        input_array_a = keras.ops.convert_to_tensor(input_array_a)
        input_array_b = keras.ops.convert_to_tensor(input_array_b)
        distances_zyx = keras.ops.convert_to_tensor(distances_zyx)

        input_array = {'input_1': input_array_a,
                       'input_2': input_array_b,
                       'input_distances': distances_zyx}

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
            experiment.links.set_link_data(patch.position_a, patch.position_b, "link_probability", float(scaled_predictions[i]))
            experiment.links.set_link_data(patch.position_a, patch.position_b, "link_penalty", float(-likelihoods[i]))


def load_link_model(model_folder: str) -> LinkModel:
    """Load a position prediction model from the given folder."""
    model_folder = os.path.abspath(model_folder)

    keras_model: keras.Model = keras.saving.load_model(os.path.join(model_folder, "model.keras"))

    # Set relevant parameters
    if not os.path.isfile(os.path.join(model_folder, "settings.json")):
        raise ValueError("Error: no settings.json found in model folder.")
    with open(os.path.join(model_folder, "settings.json")) as file_handle:
        json_contents = json.load(file_handle)
        if json_contents["type"] != "links":
            raise ValueError("Error: model at " + model_folder + " is made for working with " + str(
                json_contents["type"]) + ", not links")
        time_window = json_contents["time_window"]
        if "patch_shape_xyz" in json_contents:
            patch_shape_xyz = json_contents["patch_shape_xyz"]
            patch_shape_zyx = [patch_shape_xyz[2], patch_shape_xyz[1], patch_shape_xyz[0]]
        else:
            patch_shape_zyx = json_contents["patch_shape_zyx"]  # Seems like some versions of OrganoidTracker use this
        scaling = json_contents["platt_scaling"] if "platt_scaling" in json_contents else 1
        intercept = json_contents["platt_intercept"] if "platt_intercept" in json_contents else 0
        intercept = numpy.log10(numpy.exp(intercept))

    # Convert to tuples
    time_window = (int(time_window[0]), int(time_window[1]))
    patch_shape_zyx = (int(patch_shape_zyx[0]), int(patch_shape_zyx[1]), int(patch_shape_zyx[2]))

    return LinkModel(keras_model=keras_model,
                     time_window=time_window,
                     patch_shape_zyx=patch_shape_zyx,
                     platt_scaling=scaling,
                     platt_intercept=intercept)


def _split_into_patches(images: Images, time_point: TimePoint, links: Iterable[Tuple[Position, Position]],
                        time_window: Tuple[int, int], *,
                        patch_shape_zyx_px: Tuple[int, int, int],
                        scale_factors_zyx: Tuple[float, float, float],
                        intensity_quantiles: Tuple[float, float]) -> Iterable[_PredictionPatch]:
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

    min_intensity = float(
        numpy.min([numpy.quantile(image.array, intensity_quantiles[0]) for image in full_images.values()]))
    max_intensity = float(
        numpy.max([numpy.quantile(image.array, intensity_quantiles[1]) for image in full_images.values()]))

    # Calculate patch shape in the input image pixels (instead of the pixels the model expects)
    patch_shape_zyx_image_px = (int(patch_shape_zyx_px[0] / scale_factors_zyx[0]),
                                int(patch_shape_zyx_px[1] / scale_factors_zyx[1]),
                                int(patch_shape_zyx_px[2] / scale_factors_zyx[2]))

    # Make the patches
    for position_a, position_b in links:
        patch_array_a = _extract_patch_array_normalized(full_images, position_a, patch_shape_zyx_image_px, min_intensity, max_intensity)
        patch_array_b = _extract_patch_array_normalized(full_images, position_b, patch_shape_zyx_image_px, min_intensity, max_intensity)

        # Scale the distances the same as the images
        distance_zyx = (round((position_b.z - position_a.z) * scale_factors_zyx[0]),
                        round((position_b.y - position_a.y) * scale_factors_zyx[1]),
                        round((position_b.x - position_a.x) * scale_factors_zyx[2]))

        yield _PredictionPatch(array_a=patch_array_a, array_b=patch_array_b, position_a=position_a, position_b=position_b, distance_zyx_px=distance_zyx)


def _extract_patch_array_normalized(full_images: Dict[TimePoint, Image], position: Position ,patch_shape_zyx_image_px: Tuple[int, int, int],
                                    min_intensity: float, max_intensity: float) -> numpy.ndarray:
    """Extract a normalized patch around the given position from the full images."""
    z_start = int(round(position.z - patch_shape_zyx_image_px[0] / 2))
    y_start = int(round(position.y - patch_shape_zyx_image_px[1] / 2))
    x_start = int(round(position.x - patch_shape_zyx_image_px[2] / 2))

    array = extract_patch_array(full_images, (z_start, y_start, x_start), patch_shape_zyx_image_px)

    # Normalize patch
    array /= (max_intensity - min_intensity)
    array -= min_intensity / (max_intensity - min_intensity)
    numpy.clip(array, 0.0, 1.0, out=array)

    return array