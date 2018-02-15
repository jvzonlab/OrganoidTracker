from imaging.visualizer import Visualizer, activate
from imaging import Experiment, Frame, Particle
from matplotlib.figure import Figure
from matplotlib.backend_bases import KeyEvent
from numpy import ndarray
from networkx import Graph
from typing import Optional, Iterable
import matplotlib.pyplot as plt


def show(experiment : Experiment):
    """Creates a standard visualizer for an experiment."""
    figure = plt.figure(figsize=(8,8))
    visualizer = StandardImageVisualizer(experiment, figure)
    activate(visualizer)


class AbstractImageVisualizer(Visualizer):
    """A generic image visualizer."""

    _frame: Frame
    _frame_images: ndarray
    _z: int

    def __init__(self, experiment: Experiment, figure: Figure, frame_number: int = 1, z: int = 14):
        super().__init__(experiment, figure)

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
        self.draw_extra()
        plt.title(self.get_title(errors))

        plt.draw()

    def get_title(self, errors: int) -> str:
        title = "Frame " + str(self._frame.frame_number()) + "    (z=" + str(self._z) + ")"
        if errors != 0:
            title += " (errors: " + str(errors) + ")"
        return title

    def draw_extra(self):
        pass # Subclasses can override this

    def draw_particles(self) -> int:
        """Draws particles and links. Returns the amount of logical inconsistencies in the iamge"""

        # Draw particles
        self._draw_particles_of_frame(self._frame, marker_size=7)
        try:
            self._draw_particles_of_frame(self._experiment.get_next_frame(self._frame), color='orange')
        except KeyError:
            pass
        try:
            self._draw_particles_of_frame(self._experiment.get_previous_frame(self._frame), color='darkred')
        except KeyError:
            pass

        # Draw links
        errors = 0
        for particle in self._frame.particles():
            if abs(particle.z - self._z) > 3:
                continue
            errors += self._draw_links(particle)

        return errors

    def _draw_particles_of_frame(self, frame: Frame, color: str = 'red', marker_size:int = 6):
        for particle in frame.particles():
            dz = abs(particle.z - self._z)
            if dz > 3:
                continue

            # Draw the particle itself (as a square or circle, depending on its depth)
            marker_style = 's'
            current_marker_size = marker_size - dz
            if particle.z != self._z:
                marker_style = 'o'
            plt.plot(particle.x, particle.y, marker_style, color=color, markeredgecolor='black',
                     markersize=current_marker_size, markeredgewidth=1)

    def _draw_links(self, particle: Particle) -> int:
        """Draws links between the particles. Returns 1 if there is 1 error: the baseline links don't match the actual
        links.
        """
        links_normal = self._get_links(self._experiment.particle_links_scratch(), particle)
        links_baseline = self._get_links(self._experiment.particle_links(), particle)

        self._draw_given_links(particle, links_normal, line_style='dotted', line_width=3)
        self._draw_given_links(particle, links_baseline)

        # Check for errors
        if self._experiment.particle_links_scratch() is not None and self._experiment.particle_links() is not None:
            if links_baseline != links_normal:
                return 1
        return 0

    @staticmethod
    def _draw_given_links(particle, links, line_style='solid', line_width=1):
        for linked_particle in links:
            if linked_particle.frame_number() < particle.frame_number():
                # Drawing to past

                plt.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], color='darkred',
                         linestyle=line_style, linewidth=line_width)
            else:
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

    def _on_command(self, command: str) -> bool:
        if command[0] == "f":
            frame_str = command[1:]
            try:
                new_frame_number = int(frame_str.strip())
                self._frame, self._frame_images = self.load_frame(new_frame_number)
                self.draw_view()
            except KeyError:
                self.update_status("Unknown frame: " + frame_str)
            except ValueError:
                self.update_status("Cannot read number: " + frame_str)
            return True
        return False

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


class StandardImageVisualizer(AbstractImageVisualizer):
    """Shows microscopy images with cells and cell trajectories drawn on top.
    Left/right keys: move in time
    Up/down keys: move in z-direction
    T key: view trajectory of cell at mouse, M key: view images of mother cells
    L key: manual linking interface"""

    def __init__(self, experiment: Experiment, figure: Figure, frame_number: int = 1, z: int = 14):
        super().__init__(experiment, figure, frame_number=frame_number, z=z)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            particle = self.get_closest_particle(self._frame.particles(), event.xdata, event.ydata, self._z, 20)
            if particle is not None:
                from linking_analysis.track_visualizer import TrackVisualizer
                track_visualizer = TrackVisualizer(self._experiment, self._fig, particle)
                activate(track_visualizer)
        elif event.key == "m":
            particle = self.get_closest_particle(self._frame.particles(), event.xdata, event.ydata, self._z)
            if particle is not None:
                from linking_analysis.cell_division_visualizer import CellDivisionVisualizer
                track_visualizer = CellDivisionVisualizer(self._experiment, self._fig, particle)
                activate(track_visualizer)
        elif event.key == "l":
            from linking_analysis.link_editor import LinkEditor
            link_editor = LinkEditor(self._experiment, self._fig, frame_number=self._frame.frame_number(), z=self._z)
            activate(link_editor)
        else:
            super()._on_key_press(event)
