import os
from typing import Tuple, NamedTuple

import keras.callbacks
import tifffile
import torch
from numpy import ndarray
from torch.utils.data import DataLoader

from organoid_tracker.neural_network.position_detection_cnn.custom_filters import distance_map


class ExampleDataset(NamedTuple):
    input: ndarray  # shape: (n_samples, z, y, x, channels)
    y_true: ndarray  # shape: (n_samples, z, y, x, 1)


class WriteExamplesCallback(keras.callbacks.Callback):

    _output_folder: str
    _examples_dataset: ExampleDataset

    def __init__(self, output_folder: str, examples_dataset: ExampleDataset):
        super().__init__()
        self._output_folder = output_folder
        self._examples_dataset = examples_dataset

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            # Start of training, record initial state
            self._write_metrics("epoch_1_start", True)

    def on_epoch_end(self, epoch, logs=None):
        self._write_metrics("epoch_" + str(epoch + 1) + "_end", False)

    def _write_metrics(self, time_name: str, first_time: bool):
        predicted = keras.ops.convert_to_numpy(self.model.predict(self._examples_dataset.input))

        label = keras.ops.convert_to_tensor(self._examples_dataset.y_true)
        # dilation = keras.ops.max_pool(label, pool_size=[1, 3, 3], strides=1, padding='same')
        dilation = torch.nn.functional.max_pool3d(label, (1, 3, 3), stride=1, padding=(0, 1, 1))
        peaks = keras.ops.where(dilation == label, 1., 0)
        y_true = keras.ops.where(label > 0.1, peaks, 0)
        label, weights = distance_map(y_true)

        label = keras.ops.convert_to_numpy(label)

        for i in range(predicted.shape[0]):
            folder = os.path.join(self._output_folder, f"Example {i + 1}")
            if first_time:
                os.makedirs(folder, exist_ok=True)
                tifffile.imwrite(os.path.join(folder , f"input.tif"),
                                 self._examples_dataset.input[i, ..., 0])
                tifffile.imwrite(os.path.join(folder, f"ground_truth.tif"),
                                 label[i, ..., 0])
            tifffile.imwrite(os.path.join(folder, f"predicted_{time_name}.tif"),
                             predicted[i, ..., 0])

