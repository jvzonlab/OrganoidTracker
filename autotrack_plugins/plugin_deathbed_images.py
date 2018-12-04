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
from autotrack.imaging.maximum_intensity_image import MaximumIntensityProjector
from autotrack.linking_analysis import linking_markers

_WIDTH = 50
_HEIGHT = 50


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "View/Tracks-Deathbed images...": lambda: _generate_deathbed_images(window),
    }


def _generate_deathbed_images(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("Deathbed images", "No links found. Therefore, we cannot find cell deaths.")

    steps_back = dialog.prompt_int("Deathbed images", "How many time steps do you want to look before the cell was "
                                                      "considered dead?\nUse 0 to look at the first moment when the "
                                                      "cell was considered dead.", min=0)
    if steps_back is None:
        return
    output_file = dialog.prompt_save_file("Deathbed images", [("TIFF file", "*.tif")])
    if output_file is None:
        return

    particles = list()
    for particle in linking_markers.find_dead_particles(experiment.links):
        particle = _find_ancestor(particle, experiment.links, steps_back)
        if particle is not None:
            particles.append(particle)

    if len(particles) == 0:
        raise UserError("Deathbed images", "No deathbed images found. Are there no cell deaths, or is the number of"
                                           " steps that you want to look into the past too high?")

    window.get_application().scheduler.add_task(_ImageGeneratingTask(experiment.image_loader(), particles, output_file))


class _ImageGeneratingTask(Task):

    _particles: List[Particle]
    _image_loader: ImageLoader()
    _output_file: str

    def __init__(self, image_loader: ImageLoader, particles: List[Particle], output_file: str):
        self._image_loader = image_loader.copy()  # We will use this image loader on another thread
        self._particles = particles
        self._output_file = output_file

    def compute(self) -> bool:
        projector = MaximumIntensityProjector(self._image_loader, 5)
        out_array = numpy.zeros((len(self._particles), _HEIGHT, _WIDTH), dtype=numpy.uint16)

        for i, particle in enumerate(self._particles):
            projector.create_projection(particle, out_array[i])

        tifffile.imsave(self._output_file, out_array, compress=6)
        return True

    def on_finished(self, result: bool):
        dialog.popup_message("Death bed", "Images generated successfully!")


def _find_ancestor(particle: Particle, links: ParticleLinks, steps_back: int) -> Optional[Particle]:
    for i in range(steps_back):
        previous = links.find_pasts(particle)
        if len(previous) == 0:
            return None
        particle = previous.pop()
    return particle
