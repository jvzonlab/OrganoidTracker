from imaging.visualizer import Visualizer, activate
from imaging import Experiment, Frame, Particle
from matplotlib.figure import Figure, Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from numpy import ndarray
from networkx import Graph
from typing import Optional, Iterable
import matplotlib.pyplot as plt


def show(experiment : Experiment):
    """Creates a standard visualizer for an experiment."""
    figure = plt.figure(figsize=(8,8))
    visualizer = ImageVisualizer(experiment, figure)
    activate(visualizer)


class ImageVisualizer(Visualizer):
    """Visualizer focused on images, with short trajectories drawn on top"""

    _frame: Frame
    _frame_images: ndarray
    _z: int

    def __init__(self, experiment: Experiment, figure: Figure, frame_number: int = 1, z: int = 14):
        super(ImageVisualizer, self).__init__(experiment, figure)

        self._z = int(z)
        self._frame, self._frame_images = self.load_frame(frame_number)

    def load_frame(self, frame_number: int):
        frame = self._experiment.get_frame(frame_number)
        frame_images = frame.load_images()

        return frame, frame_images

    def draw_view(self):
        self._clear_axis()
        if self._frame_images is not None:
            image = self._ax.imshow(self._frame_images[self._z], cmap="gray")
            plt.colorbar(mappable=image, ax=self._ax)
        errors = self.draw_particles()
        plt.title(self.get_title(errors))

        plt.draw()

    def get_title(self, errors: int) -> str:
        title = "Frame " + str(self._frame.frame_number()) + "    (z=" + str(self._z) + ")"
        if errors != 0:
            title += " (errors: " + str(errors) + ")"
        return title

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

        self._draw_given_links(particle, links_normal, marker_size=marker_size, marker_style=marker_style,
                               line_style='dotted', line_width=3)
        self._draw_given_links(particle, links_baseline, marker_size=marker_size, marker_style=marker_style)

        # Check for errors
        if self._experiment.particle_links() is not None and self._experiment.particle_links_baseline() is not None:
            if links_baseline != links_normal:
                return 1
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
                plt.plot(linked_particle.x, linked_particle.y, marker_style, color='orange', markeredgecolor='black',
                         markersize=marker_size, markeredgewidth=1)
                plt.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], color='orange',
                         linestyle=line_style, linewidth=line_width)

    def _get_links(self, network: Optional[Graph], particle: Particle) -> Iterable[Particle]:
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
        elif event.key == "t":
            particle = self.get_closest_particle(self._frame.particles(), event.xdata, event.ydata, self._z, 20)
            if particle is not None:
                from imaging.track_visualizer import TrackVisualizer
                track_visualizer = TrackVisualizer(self._experiment, self._fig, particle)
                activate(track_visualizer)

    def _on_command(self, command: str):
        if command[0] == "f":
            frame_str = command[1:]
            try:
                new_frame_number = int(frame_str.strip())
                self._frame, self._frame_images = self.load_frame(new_frame_number)
                self.draw_view()
            except KeyError:
                print("Unknown frame: " + frame_str)
            except ValueError:
                print("Cannot read number: " + frame_str)
            return
        print("Unknown command: " + command)

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


