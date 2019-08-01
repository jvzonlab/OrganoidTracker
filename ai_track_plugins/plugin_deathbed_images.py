import cv2
import os
from os import path

import tifffile
from typing import Any, List, Optional, Dict

import numpy

from ai_track.core import UserError, clamp
from ai_track.core.spline import SplineCollection, DataAxisPosition
from ai_track.core.image_loader import ImageLoader, ImageChannel
from ai_track.core.images import Images
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.gui import dialog
from ai_track.gui.threading import Task
from ai_track.gui.window import Window
from ai_track.imaging import cropper, bits
from ai_track.linking_analysis import linking_markers, position_connection_finder

_MARGIN = 40


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Cell deaths//Deathbed images...": lambda: _generate_deathbed_images(window),
    }


class _Deathbed:
    """Represents the final positions of a position before it died."""

    positions: List[Position]  # List of positions. Death at index 0, one time point before at index 1, etc.
    axis_positions: List[Optional[DataAxisPosition]]  # Same list, but now on the data axis

    def __init__(self, positions: List[Position], axis_positions: List[Optional[DataAxisPosition]]):
        self.positions = positions
        self.axis_positions = axis_positions

    def image_min_x(self) -> int:
        return int(min((position.x for position in self.positions))) - _MARGIN

    def image_max_x(self) -> int:
        return int(max((position.x for position in self.positions))) + _MARGIN

    def image_min_y(self) -> int:
        return int(min((position.y for position in self.positions))) - _MARGIN

    def image_max_y(self) -> int:
        return int(max((position.y for position in self.positions))) + _MARGIN

    def get_file_name(self, suffix: str) -> str:
        death_position = self.positions[0]
        death_axis_position = self.axis_positions[0]
        death_axis_str = f"a{death_axis_position.pos:.0f}-" if death_axis_position is not None else ""
        return f"{death_axis_str}x{death_position.x:.0f}-y{death_position.y:.0f}-z{death_position.z:.0f}" \
            f"-t{death_position.time_point_number()}{suffix}"


def _generate_deathbed_images(window: Window):
    experiment = window.get_experiment()
    data_axes = experiment.splines
    links = experiment.links
    if not links.has_links():
        raise UserError("Deathbed images", "No links found. Therefore, we cannot find cell deaths.")

    steps_back = dialog.prompt_int("Deathbed images", "How many time steps do you want to look before the cell was "
                                                      "considered dead?\nUse 0 to look at only the first moment when "
                                                      "the cell was considered dead.", minimum=0)
    if steps_back is None:
        return

    deathbeds = list()
    for position in linking_markers.find_death_and_shed_positions(links):
        before_death_list = position_connection_finder.find_previous_positions(position, links, steps_back)

        if before_death_list is None:
            continue

        complete_position_list = [position] + before_death_list
        complete_axis_position_list = [data_axes.to_position_on_original_axis(links, pos)
                                       for pos in complete_position_list]
        deathbeds.append(_Deathbed(complete_position_list, complete_axis_position_list))

    if len(deathbeds) == 0:
        raise UserError("Deathbed images", "No deathbed images found. Are there no cell deaths, or is the number of"
                                           " steps that you want to look into the past too high?")

    output_folder = dialog.prompt_save_file("Deathbed images", [("Folder", "*")])
    if output_folder is None:
        return
    if path.exists(output_folder):
        raise UserError("Deathbed images",
                        f"A file already exists at {output_folder}. Therefore, we cannot create a directory there.")
    os.mkdir(output_folder)

    window.get_scheduler().add_task(_ImageGeneratingTask(experiment.images, deathbeds, output_folder))


class _ImageGeneratingTask(Task):

    _deathbeds: List[_Deathbed]
    _images: Images
    _output_folder: str

    def __init__(self, images: Images, deathbeds: List[_Deathbed], output_folder: str):
        self._images = images.copy()  # We will use this image loader on another thread
        self._deathbeds = deathbeds
        self._output_folder = output_folder

    def compute(self) -> bool:
        for deathbed in self._deathbeds:
            min_x, max_x = deathbed.image_min_x(), deathbed.image_max_x()
            min_y, max_y = deathbed.image_min_y(), deathbed.image_max_y()

            out_array = numpy.zeros((len(deathbed.positions), max_y - min_y, max_x - min_x, 3), dtype=numpy.uint8)

            for i in range(len(deathbed.positions)):
                image_index = len(deathbed.positions) - i - 1
                position = deathbed.positions[i]
                image = self._images.get_image(position.time_point())
                image.array = bits.image_to_8bit(image.array)
                z = clamp(image.min_z, int(position.z), image.limit_z - 1)

                cropper.crop_2d(image, min_x, min_y, z, out_array[image_index, :, :, 0])
                out_array[image_index, :, :, 1] = out_array[image_index, :, :, 0]  # Update green channel too
                out_array[image_index, :, :, 2] = out_array[image_index, :, :, 0]  # Update blue channel too

                local_pos = int(position.x - min_x), int(position.y - min_y)
                cv2.circle(out_array[image_index, :, :], local_pos, 3, (255, 0, 0), cv2.FILLED)

            tifffile.imsave(path.join(self._output_folder, deathbed.get_file_name(".tif")), out_array, compress=6)
        return True

    def on_finished(self, result: bool):
        dialog.popup_message("Death bed", "Images generated successfully!")

