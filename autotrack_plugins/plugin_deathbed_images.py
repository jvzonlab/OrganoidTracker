import cv2
import os
from os import path

import tifffile
from typing import Any, List, Optional, Dict

import numpy

from autotrack.core import UserError
from autotrack.core.image_loader import ImageLoader
from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.gui import dialog
from autotrack.gui.threading import Task
from autotrack.gui.window import Window
from autotrack.imaging import cropper, bits
from autotrack.linking_analysis import linking_markers

_MARGIN = 40


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "View/Cell deaths-Cell deaths/Deathbed images...": lambda: _generate_deathbed_images(window),
    }


class _Deathbed:
    """Represents the final positions of a particle before it died."""

    particles: List[Particle]  # List of particles. Death at index 0, one time point before at index 1, etc.

    def __init__(self, first: Particle):
        self.particles = [first]

    def image_min_x(self) -> int:
        return int(min((particle.x for particle in self.particles))) - _MARGIN

    def image_max_x(self) -> int:
        return int(max((particle.x for particle in self.particles))) + _MARGIN

    def image_min_y(self) -> int:
        return int(min((particle.y for particle in self.particles))) - _MARGIN

    def image_max_y(self) -> int:
        return int(max((particle.y for particle in self.particles))) + _MARGIN

    def get_file_name(self, suffix: str) -> str:
        death_particle = self.particles[0]
        return f"x{death_particle.x:.0f}-y{death_particle.y:.0f}-z{death_particle.z:.0f}" \
            f"-t{death_particle.time_point_number()}{suffix}"


def _generate_deathbed_images(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("Deathbed images", "No links found. Therefore, we cannot find cell deaths.")

    steps_back = dialog.prompt_int("Deathbed images", "How many time steps do you want to look before the cell was "
                                                      "considered dead?\nUse 0 to look at only the first moment when "
                                                      "the cell was considered dead.", min=0)
    if steps_back is None:
        return

    particles = list()
    for particle in linking_markers.find_dead_particles(experiment.links):
        particle = _find_ancestor(particle, experiment.links, steps_back)
        if particle is not None:
            particles.append(particle)

    if len(particles) == 0:
        raise UserError("Deathbed images", "No deathbed images found. Are there no cell deaths, or is the number of"
                                           " steps that you want to look into the past too high?")

    output_folder = dialog.prompt_save_file("Deathbed images", [("Folder", "*")])
    if output_folder is None:
        return
    if path.exists(output_folder):
        raise UserError("Deathbed images",
                        f"A file already exists at {output_folder}. Therefore, we cannot create a directory there.")
    os.mkdir(output_folder)

    window.get_scheduler().add_task(_ImageGeneratingTask(experiment.image_loader(), particles, output_folder))


class _ImageGeneratingTask(Task):

    _deathbeds: List[_Deathbed]
    _image_loader: ImageLoader()
    _output_folder: str

    def __init__(self, image_loader: ImageLoader, deathbeds: List[_Deathbed], output_folder: str):
        self._image_loader = image_loader.copy()  # We will use this image loader on another thread
        self._deathbeds = deathbeds
        self._output_folder = output_folder

    def compute(self) -> bool:
        for deathbed in self._deathbeds:
            min_x, max_x = deathbed.image_min_x(), deathbed.image_max_x()
            min_y, max_y = deathbed.image_min_y(), deathbed.image_max_y()

            out_array = numpy.zeros((len(deathbed.particles), max_y - min_y, max_x - min_x, 3), dtype=numpy.uint8)

            for i in range(len(deathbed.particles)):
                image_index = len(deathbed.particles) - i - 1
                particle = deathbed.particles[i]
                image = bits.image_to_8bit(self._image_loader.get_image_stack(particle.time_point()))
                z = min(max(0, int(particle.z)), len(image) - 1)

                cropper.crop_2d(image, min_x, min_y, z, out_array[image_index, :, :, 0])
                out_array[image_index, :, :, 1] = out_array[image_index, :, :, 0]  # Update green channel too
                out_array[image_index, :, :, 2] = out_array[image_index, :, :, 0]  # Update blue channel too

                local_pos = int(particle.x - min_x), int(particle.y - min_y)
                cv2.circle(out_array[image_index, :, :], local_pos, 3, (255, 0, 0), cv2.FILLED)

            tifffile.imsave(path.join(self._output_folder, deathbed.get_file_name(".tif")), out_array, compress=6)
        return True

    def on_finished(self, result: bool):
        dialog.popup_message("Death bed", "Images generated successfully!")


def _find_ancestor(particle: Particle, links: ParticleLinks, steps_back: int) -> Optional[_Deathbed]:
    deathbed = _Deathbed(particle)
    for i in range(steps_back):
        previous = links.find_pasts(particle)
        if len(previous) == 0:
            return None
        particle = previous.pop()
        deathbed.particles.append(particle)
    return deathbed
