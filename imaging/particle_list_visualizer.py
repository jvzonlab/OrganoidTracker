from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from imaging import Particle, Experiment
from imaging.visualizer import Visualizer, activate
from networkx import Graph
from typing import List, Optional
import matplotlib.pyplot as plt
import imaging


class ParticleListVisualizer(Visualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    _current_particle_index: int
    _particle_list = List[Particle]
    _show_next_image: bool

    __last_index = -1  # Static variable

    def __init__(self, experiment: Experiment, figure: Figure, all_particles: List[Particle],
                 chosen_particle: Optional[Particle] = None, show_next_image: bool = False):
        """Creates a viewer for a list of particles. The particles will automatically be sorted by time_point number.
        chosen_particle is a particle that is used as a starting point for the viewer, but only if it appears in the
        list
        """
        super().__init__(experiment, figure)
        self._particle_list = all_particles
        self._particle_list.sort(key=lambda particle: particle.time_point_number())
        self._current_particle_index = self._find_closest_particle_index(chosen_particle)
        self._show_next_image = show_next_image

    def _find_closest_particle_index(self, particle: Optional[Particle]) -> int:
        if particle is None:
            return ParticleListVisualizer.__last_index  # Give up immediately
        try:
            return self._particle_list.index(particle)
        except ValueError:
            # Try nearest particle
            close_match = imaging.get_closest_particle(self._particle_list, particle, max_distance=200)

            if close_match is not None:
                return self._particle_list.index(close_match)
            return ParticleListVisualizer.__last_index  # Give up

    def get_message_no_particles(self):
        return "No cells found. Is there some data missing?"

    def get_message_press_right(self):
        return "Press right to view the first cell."

    def draw_view(self):
        self._clear_axis()
        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            if len(self._particle_list) == 0:
                plt.title(self.get_message_no_particles())
            else:
                plt.title(self.get_message_press_right())
            plt.draw()
            return

        self._zoom_to_cell()
        self._show_image()

        current_particle = self._particle_list[self._current_particle_index]
        self._draw_particle(current_particle)
        self._draw_connections(self._experiment.particle_links(), current_particle)
        self._draw_connections(self._experiment.particle_links_scratch(), current_particle, line_style='dotted',
                               line_width=3)
        plt.title(self.get_title(self._particle_list, self._current_particle_index))

        plt.draw()
        ParticleListVisualizer.__last_index = self._current_particle_index

    def _zoom_to_cell(self):
        mother = self._particle_list[self._current_particle_index]
        self._ax.set_xlim(mother.x - 50, mother.x + 50)
        self._ax.set_ylim(mother.y + 50, mother.y - 50)
        self._ax.set_autoscale_on(False)

    def _draw_particle(self, particle: Particle, color=imaging.COLOR_CELL_CURRENT, size=7):
        style = 's'
        dz = abs(particle.time_point_number() - self._particle_list[self._current_particle_index].time_point_number())
        if dz != 0:
            style='o'
            size -= dz
        self._ax.plot(particle.x, particle.y, style, color=color, markeredgecolor='black', markersize=size,
                      markeredgewidth=1)

    def _draw_connections(self, graph: Optional[Graph], main_particle: Particle, line_style:str = "solid",
                          line_width:int = 1):
        if graph is None:
            return
        try:
            for connected_particle in graph[main_particle]:
                color = imaging.COLOR_CELL_NEXT
                if connected_particle.time_point_number() < main_particle.time_point_number():
                    color = imaging.COLOR_CELL_PREVIOUS
                    if self._show_next_image:
                        continue  # Showing the previous position only makes things more confusing here
                self._ax.plot([connected_particle.x, main_particle.x], [connected_particle.y, main_particle.y],
                              color=color, linestyle=line_style, linewidth=line_width)
                self._draw_particle(connected_particle, color=color, size=6)
        except KeyError:
            pass

    def _show_image(self):
        mother = self._particle_list[self._current_particle_index]
        time_point = self._experiment.get_time_point(mother.time_point_number())
        image_stack = self.create_image(time_point, self._show_next_image)
        if image_stack is not None:
            image = self._ax.imshow(image_stack[int(mother.z)], cmap="gray")
            if not self._show_next_image:
                plt.colorbar(mappable=image, ax=self._ax)

    def _goto_next(self):
        self._current_particle_index += 1
        if self._current_particle_index >= len(self._particle_list):
            self._current_particle_index = 0
        self.draw_view()

    def _goto_previous(self):
        self._current_particle_index -= 1
        if self._current_particle_index < 0:
            self._current_particle_index = len(self._particle_list) - 1
        self.draw_view()

    def goto_full_image(self):
        from imaging.image_visualizer import StandardImageVisualizer

        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            # Don't know where to go
            image_visualizer = StandardImageVisualizer(self._experiment, self._fig,
                                                       show_next_image=self._show_next_image)
        else:
            mother = self._particle_list[self._current_particle_index]
            image_visualizer = StandardImageVisualizer(self._experiment, self._fig,
                                                       time_point_number=mother.time_point_number(), z=int(mother.z),
                                                       show_next_image=self._show_next_image)
        activate(image_visualizer)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "left":
            self._goto_previous()
        elif event.key == "right":
            self._goto_next()
        elif event.key == imaging.KEY_SHOW_NEXT_IMAGE_ON_TOP:
            self._show_next_image = not self._show_next_image
            self.draw_view()

    def get_title(self, _all_mothers: List[Particle], _mother_index: int):
        mother = _all_mothers[_mother_index]
        return "Cell " + str(self._current_particle_index + 1) + "/" + str(len(self._particle_list)) + "\n" + str(mother)
