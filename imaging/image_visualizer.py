import imaging
from imaging.visualizer import Visualizer, activate
from imaging import Experiment, TimePoint, Particle
from matplotlib.figure import Figure
from matplotlib.backend_bases import KeyEvent
from numpy import ndarray
from networkx import Graph
from typing import Optional, Iterable, List, Tuple
import matplotlib.pyplot as plt
import numpy


def show(experiment : Experiment):
    """Creates a standard visualizer for an experiment."""
    figure = plt.figure(figsize=(8,8))
    visualizer = StandardImageVisualizer(experiment, figure)
    activate(visualizer)


class AbstractImageVisualizer(Visualizer):
    """A generic image visualizer."""

    MAX_Z_DISTANCE: int = 3

    _time_point: TimePoint
    _time_point_images: ndarray
    _z: int
    __drawn_particles: List[Particle]
    _drawn_time_point_images: ndarray
    _show_next_image: bool = False

    def __init__(self, experiment: Experiment, figure: Figure, time_point_number: Optional[int] = None, z: int = 14,
                 show_next_image: bool = False):
        super().__init__(experiment, figure)

        if time_point_number is None:
            time_point_number = experiment.first_time_point_number()
        self._z = int(z)
        self._show_next_image = show_next_image
        self._time_point, self._time_point_images = self.load_time_point(time_point_number)
        self.__drawn_particles = []

    def load_time_point(self, time_point_number: int) -> Tuple[TimePoint, ndarray]:
        time_point = self._experiment.get_time_point(time_point_number)
        time_point_images = self.create_image(time_point, self._show_next_image)

        return time_point, time_point_images

    def draw_view(self):
        self._clear_axis()
        self.__drawn_particles.clear()
        self._draw_image()
        errors = self.draw_particles()
        self.draw_extra()
        plt.title(self.get_title(errors))

        plt.draw()

    def _draw_image(self):
        if self._time_point_images is not None:
            image = self._ax.imshow(self._time_point_images[self._z], cmap="gray")
            if not self._show_next_image:
                plt.colorbar(mappable=image, ax=self._ax)

    def get_title(self, errors: int) -> str:
        title = "Time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"
        if errors != 0:
            title += " (changes: " + str(errors) + ")"
        return title

    def draw_extra(self):
        pass # Subclasses can override this

    def draw_particles(self) -> int:
        """Draws particles and links. Returns the amount of non-equal links in the image"""

        # Draw particles
        self._draw_particles_of_time_point(self._time_point, marker_size=7)

        # Next time point
        has_linking_data = self._experiment.particle_links() is not None \
                           or self._experiment.particle_links_scratch() is not None
        if self._show_next_image or has_linking_data:
            # Only draw particles of next/previous time point if there is linking data
            try:
                self._draw_particles_of_time_point(self._experiment.get_next_time_point(self._time_point), color='red')
            except KeyError:
                pass  # There is no next time point, ignore

        # Previous time point
        if not self._show_next_image and has_linking_data:
            try:
                self._draw_particles_of_time_point(self._experiment.get_previous_time_point(self._time_point),
                                                   color='blue')
            except KeyError:
                pass  # There is no previous time point, ignore

        # Draw links
        errors = 0
        for particle in self._time_point.particles():
            errors += self._draw_links(particle)

        return errors

    def _draw_particles_of_time_point(self, time_point: TimePoint, color: str = imaging.COLOR_CELL_CURRENT,
                                      marker_size:int = 6):
        for particle in time_point.particles():
            dz = abs(particle.z - self._z)
            if dz > self.MAX_Z_DISTANCE:
                continue

            # Draw the particle itself (as a square or circle, depending on its depth)
            marker_style = 's'
            current_marker_size = marker_size - dz
            if int(particle.z) != self._z:
                marker_style = 'o'
            self._draw_particle(particle, color, current_marker_size, marker_style)

    def _draw_particle(self, particle, color, current_marker_size, marker_style):
        # Draw color marker
        marker_color = self._experiment.marker().get_color(self._time_point, particle)
        if marker_color is not None:
            plt.plot(particle.x, particle.y, "o", color=marker_color, markersize=current_marker_size * 3,
                     markeredgecolor='black', markeredgewidth=1)

        # Draw error marker
        graph = self._experiment.particle_links_scratch() or self._experiment.particle_links()
        if graph is not None and particle in graph and "error" in graph.nodes[particle]:
            plt.plot(particle.x, particle.y, 'X', color='black', markeredgecolor='white',
                 markersize=current_marker_size + 12, markeredgewidth=2)

        # Draw particle
        plt.plot(particle.x, particle.y, marker_style, color=color, markeredgecolor='black',
                 markersize=current_marker_size, markeredgewidth=1)
        self.__drawn_particles.append(particle)

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

    def _draw_given_links(self, particle, links, line_style='solid', line_width=1):
        for linked_particle in links:
            if abs(linked_particle.z - self._z) > self.MAX_Z_DISTANCE\
                    and abs(particle.z - self._z) > self.MAX_Z_DISTANCE:
                continue
            if linked_particle.time_point_number() < particle.time_point_number():
                # Drawing to past
                if not self._show_next_image:
                    plt.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], linestyle=line_style,
                             color=imaging.COLOR_CELL_PREVIOUS, linewidth=line_width)
            else:
                plt.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], linestyle=line_style,
                         color=imaging.COLOR_CELL_NEXT, linewidth=line_width)

    def _get_links(self, network: Optional[Graph], particle: Particle) -> Iterable[Particle]:
        if network is None:
            return []
        try:
            return network[particle]
        except KeyError:
            return []

    def _get_particle_at(self, x: Optional[int], y: Optional[int]) -> Optional[Particle]:
        """Wrapper of get_closest_particle that makes use of the fact that we can lookup all particles ourselves."""
        return self.get_closest_particle(self.__drawn_particles, x, y, None, max_distance=5)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "up":
            self._move_in_z(1)
        elif event.key == "down":
            self._move_in_z(-1)
        elif event.key == "left":
            self._move_in_time(-1)
        elif event.key == "right":
            self._move_in_time(1)
        elif event.key == imaging.KEY_SHOW_NEXT_IMAGE_ON_TOP:
            self._show_next_image = not self._show_next_image
            self._move_in_time(0)  # Reloads images

    def _on_command(self, command: str) -> bool:
        if command[0] == "t":
            time_point_str = command[1:]
            try:
                new_time_point_number = int(time_point_str.strip())
                self._time_point, self._time_point_images = self.load_time_point(new_time_point_number)
                self.draw_view()
            except KeyError:
                self.update_status("Unknown time point: " + time_point_str + " (range is "
                                   + str(self._experiment.first_time_point_number()) + " to "
                                   + str(self._experiment.last_time_point_number()) + ", inclusive)")
            except ValueError:
                self.update_status("Cannot read number: " + time_point_str)
            return True
        if command == "help":
            self.update_status("/t20: Jump to time point 20 (also works for other time points)")
            return True
        return False

    def _move_in_z(self, dz: int):
        old_z = self._z
        self._z += dz

        if self._z < 0:
            self._z = 0
        if self._z >= len(self._time_point_images):
            self._z = len(self._time_point_images) - 1

        if self._z != old_z:
            self.draw_view()

    def _move_in_time(self, dt: int):
        old_time_point_number = self._time_point.time_point_number()
        new_time_point_number = old_time_point_number + dt
        try:
            self._time_point, self._time_point_images = self.load_time_point(new_time_point_number)
            self.draw_view()
        except KeyError:
            pass


class StandardImageVisualizer(AbstractImageVisualizer):
    """Cell and image viewer

    Moving: left/right moves in time, up/down in the z-direction and type '/t30' + ENTER to jump to time point 30
    Viewing: N shows next frame in red, current in green and T shows trajectory of cell under the mouse cursor
    Cell lists: M shows mother cells, E shows detected errors and D shows differences between two loaded data sets
    Editing: L shows an editor for links"""

    def __init__(self, experiment: Experiment, figure: Figure, time_point_number: Optional[int] = None, z: int = 14,
                 show_next_image: bool = False):
        super().__init__(experiment, figure, time_point_number=time_point_number, z=z, show_next_image=show_next_image)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is not None:
                from linking_analysis.track_visualizer import TrackVisualizer
                track_visualizer = TrackVisualizer(self._experiment, self._fig, particle)
                activate(track_visualizer)
        elif event.key == "m":
            particle = self._get_particle_at(event.xdata, event.ydata)
            from linking_analysis.cell_division_visualizer import CellDivisionVisualizer
            track_visualizer = CellDivisionVisualizer(self._experiment, self._fig, particle)
            activate(track_visualizer)
        elif event.key == "e":
            particle = self._get_particle_at(event.xdata, event.ydata)
            from imaging.errors_visualizer import ErrorsVisualizer
            warnings_visualizer = ErrorsVisualizer(self._experiment, self._fig, particle)
            activate(warnings_visualizer)
        elif event.key == "d":
            particle = self._get_particle_at(event.xdata, event.ydata)
            from linking_analysis.differences_visualizer import DifferencesVisualizer
            differences_visualizer = DifferencesVisualizer(self._experiment, self._fig, particle)
            activate(differences_visualizer)
        elif event.key == "l":
            from linking_analysis.link_editor import LinkEditor
            link_editor = LinkEditor(self._experiment, self._fig, time_point_number=self._time_point.time_point_number(), z=self._z)
            activate(link_editor)
        else:
            super()._on_key_press(event)
