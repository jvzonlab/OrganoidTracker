from imaging import Experiment, Frame, Particle
from matplotlib.figure import Figure, Axes
from matplotlib.backend_bases import KeyEvent
from numpy import ndarray
from networkx import Graph
from typing import Optional
import matplotlib.pyplot as plt
import tifffile


_visualizer = None # Reference to prevent event handler from being garbage collected


def visualize(experiment : Experiment):
    _configure_matplotlib()

    # Keep reference to avoid event handler from being garbage collected
    global _visualizer
    _visualizer = Visualizer(experiment)

def _configure_matplotlib():
    plt.rcParams['keymap.forward'] = []
    plt.rcParams['keymap.back'] = ['backspace']

class Visualizer:
    """A complete application for visualization of an experiment"""

    _experiment: Experiment

    _frame: Frame
    _frame_images: ndarray
    _z: int

    _fig: Figure
    _ax: Axes

    def __init__(self, experiment: Experiment):
        self._experiment = experiment
        self._z = 14

        self._fig = plt.figure(figsize=(8,8))
        self._ax = self._fig.gca()
        self._fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._frame, self._frame_images = self.load_frame(1)
        self.draw_view()

    def load_frame(self, frame_number: int):
        frame = self._experiment.get_frame(frame_number)
        image_file = frame.image_file_name()

        frame_images = None
        if image_file is not None:
            frame_images = _read_image_file(image_file)

        return frame, frame_images

    def draw_view(self):
        self._clear_axis()
        if self._frame_images is not None:
            self._ax.imshow(self._frame_images[self._z])
        errors = self.draw_particles()

        plt.title(self.get_title(errors))
        plt.draw()

    def get_title(self, errors: int) -> str:
        title = "Frame " + str(self._frame.frame_number()) + "    (z=" + str(self._z) + ")"
        if errors != 0:
            title += " (errors: " + str(errors) + ")"
        return title

    def _clear_axis(self):
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        self._ax.clear()
        if xlim[1] - xlim[0] > 2:
            # Only preserve scale if some sensible value was recorded
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_autoscale_on(False)

    def draw_particles(self) -> int:
        """Draws particles and links. Returns the amount of logical inconsistencies in the iamge"""
        errors = 0
        for particle in self._frame.particles():
            if abs(particle.z - self._z) > 2:
                continue
            errors += self._draw_links(particle)

            # Draw the particle itself (as a square or circle, depending on its depth)
            marker_style = 's'
            marker_size = 7
            if particle.z != self._z:
                marker_style = 'o'
                marker_size = 5
            plt.plot(particle.x, particle.y, marker_style, color='red', markeredgecolor='black', markersize=marker_size,
                     markeredgewidth=1)

        return errors

    def _draw_links(self, particle: Particle) -> int:
        """Draws links between the particles. Returns 1 if there is 1 error: the baseline links don't match the actual
        links.
        """
        links_normal = self._get_links(self._experiment.particle_links(), particle)
        links_baseline = self._get_links(self._experiment.particle_links_baseline(), particle)

        marker_style = 's'
        marker_size = 6
        if particle.z != self._z:
            marker_style = 'o'
            marker_size = 4

        if len(links_baseline) != 0 and links_normal != links_baseline:
            # Links don't match
            self._draw_given_links(particle, links_normal, marker_size=marker_size, marker_style=marker_style,
                                   line_style='dotted', line_width=3)
            self._draw_given_links(particle, links_baseline, marker_size=marker_size, marker_style=marker_style)
            return 1
        else:
            # Links do match
            self._draw_given_links(particle, links_normal, marker_size=marker_size, marker_style=marker_style)
            return 0

    @staticmethod
    def _draw_given_links(particle, links, line_style='solid', line_width=1, marker_style='s', marker_size=7):
        for linked_particle in links:
            if linked_particle.frame_number() < particle.frame_number():
                # Drawing to past

                plt.plot(linked_particle.x, linked_particle.y, marker_style, color='darkred', markeredgecolor='black',
                         markersize=marker_size, markeredgewidth=1)
                plt.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], color='darkred',
                         linestyle=line_style, linewidth=line_width)
            else:
                plt.plot(linked_particle.x, linked_particle.y, marker_style, color='pink', markeredgecolor='black',
                         markersize=marker_size, markeredgewidth=1)
                plt.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], color='pink',
                         linestyle=line_style, linewidth=line_width)

    def _get_links(self, network: Optional[Graph], particle: Particle):
        if network is None:
            return []
        try:
            return network[particle]
        except KeyError:
            return []

    def _on_key_press(self, event: KeyEvent):
        if event.key == "up":
            self._move_in_z(1)
        elif event.key == "down":
            self._move_in_z(-1)
        elif event.key == "left":
            self._move_in_time(-1)
        elif event.key == "right":
            self._move_in_time(1)

    def _move_in_z(self, dz: int):
        old_z = self._z
        self._z += dz

        if self._z < 0:
            self._z = 0
        if self._z >= len(self._frame_images):
            self._z = len(self._frame_images) - 1

        if self._z != old_z:
            self.draw_view()

    def _move_in_time(self, dt: int):
        old_frame_number = self._frame.frame_number()
        new_frame_number = old_frame_number + dt
        try:
            self._frame, self._frame_images = self.load_frame(new_frame_number)
            self.draw_view()
        except KeyError:
            pass


def _read_image_file(image_file) -> ndarray:
    with tifffile.TiffFile(image_file) as f:
        return f.asarray(maxworkers=None)
