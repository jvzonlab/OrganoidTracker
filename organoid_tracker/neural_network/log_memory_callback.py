from typing import Union

import keras
import psutil
import torch
import datetime


class LogMemoryCallback(keras.callbacks.Callback):
    """Logs the GPU memory usage at the start of training, at the end of each epoch, and at the end of every 10th
    batch."""

    _output_file: str
    _current_epoch: int = 0

    def __init__(self, output_file: str):
        super().__init__()
        self._output_file = output_file

        # Write header to the output file
        with open(self._output_file, 'w') as handle:
            handle.write("# Logs GPU and RAM memory usage after each epoch. Epoch 0 is before training starts.\n")
            handle.write("time,epoch,batch,gpu_memory_allocated_gb,gpu_memory_reserved_gb,ram_used_gb\n")

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch  # Keep track of the current epoch

        if epoch == 0:
            self._log(-1, "pre-training")  # Log before the first epoch starts

    def on_epoch_end(self, epoch, logs=None):
        self._log(epoch, "end-of-epoch")

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Log every 10 batches
            self._log(self._current_epoch, batch)

    def _log(self, epoch: int, batch: Union[int, str]):
        # Retrieve GPU memory usage
        if torch.cuda.is_available():
            gpu_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        else:
            gpu_allocated_gb = 0.0
            gpu_reserved_gb = 0.0

        # Retrieve RAM usage
        process = psutil.Process()
        memory_info = process.memory_info()
        ram_used_gb = memory_info.rss / (1024 ** 3)  # Convert to GB

        # Retrieve time in a format that Excel can parse
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self._output_file, 'a') as handle:
            handle.write(f"{current_time},{epoch + 1},{batch},{gpu_allocated_gb:.4f},{gpu_reserved_gb:.4f},{ram_used_gb:.4f}\n")
