import os
import os.path
from typing import Dict, Any

import tifffile
from tifffile import TiffWriter

from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.image_loader import ImageLoader
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.util import bits


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export movie//TIF-TIF stacks...": lambda: _export_timelapse(window)
    }


def _export_timelapse(window: Window):
    image_loader = window.get_experiment().images.image_loader()
    file = dialog.prompt_save_file("TIF folder", [("Folder", "*")])
    if file is None:
        return
    window.get_scheduler().add_task(_ImageWriteTask(image_loader, file))


class _ImageWriteTask(Task):

    _output_file: str
    _image_loader: ImageLoader

    def __init__(self, image_loader: ImageLoader, output_file: str):
        self._image_loader = image_loader.copy()
        self._output_file = output_file

    def compute(self) -> Any:
        _write_images(self._image_loader, self._output_file)
        return True

    def on_finished(self, result: Any):
        dialog.popup_message("Image created", f"Image written to {self._output_file}.")


def _write_images(image_loader: ImageLoader, output_folder: str):
    # This format can handle data types uint8, uint16, or float32 and
    # data shapes up to 6 dimensions in TZCYXS order.
    if os.path.exists(output_folder):
        os.remove(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    first_time_point_number = image_loader.first_time_point_number()
    last_time_point_number = image_loader.last_time_point_number()
    if first_time_point_number is None or last_time_point_number is None:
        raise UserError("Image time points not found", "Cannot determine first and last time point of image series. "
                                                       "Did you load any images? Cannot save anything.")
    image_size_zyx = image_loader.get_image_size_zyx()
    if image_size_zyx is None:
        raise UserError("Image time points not found", "Cannot determine first and last time point of image series. "
                                                       "Did you load any images? Cannot save anything.")
    channels = image_loader.get_channels()
    for time_point_number in range(first_time_point_number, last_time_point_number + 1, 1):
        print(f"Working on time point {time_point_number}...")
        time_point = TimePoint(time_point_number)
        for i, channel in enumerate(channels):
            image_3d = image_loader.get_3d_image_array(time_point, channel)
            if image_3d is None:
                continue
            image_3d = bits.ensure_8bit(image_3d)
            file = os.path.join(output_folder, os.path.basename(output_folder) + f"t{time_point_number:03d}c{i+1}.tif")
            tifffile.imwrite(file, image_3d, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})
