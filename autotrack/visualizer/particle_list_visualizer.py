from typing import List, Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent

from autotrack import core
from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle, get_closest_particle
from autotrack.gui.window import Window
from autotrack.visualizer import Visualizer, activate, DisplaySettings


class ParticleListVisualizer(Visualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    _current_particle_index: int = -1
    _particle_list = List[Particle]
    _show_next_image: bool

    __last_particle_by_class = dict()  # Static variable

    def __init__(self, window: Window, all_particles: List[Particle], chosen_particle: Optional[Particle] = None,
                 show_next_image: bool = False):
        """Creates a viewer for a list of particles. The particles will automatically be sorted by time_point number.
        chosen_particle is a particle that is used as a starting point for the viewer, but only if it appears in the
        list
        """
        super().__init__(window)
        self._particle_list = all_particles
        self._particle_list.sort(key=lambda particle: particle.time_point_number())
        self._show_closest_or_stored_particle(chosen_particle)  # Calling a self.method during construction is bad...
        self._show_next_image = show_next_image

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View/Exit-Exit this view (Esc)": lambda: self._exit_view(),
            "Navigate/Time-Next (Right)": self._goto_next,
            "Navigate/Time-Previous (Left)": self._goto_previous
        }

    def _show_closest_or_stored_particle(self, particle: Optional[Particle]):
        if particle is not None:
            # Try to find selected particle
            try:
                self._current_particle_index = self._particle_list.index(particle)
            except ValueError:
                # Try nearest particle
                close_match = get_closest_particle(self._particle_list, particle, max_distance=100)

                if close_match is not None and close_match.time_point_number() == particle.time_point_number():
                    self._current_particle_index = self._particle_list.index(close_match)
                    return

        # Give up, show particle from before
        try:
            particle = self._get_last_particle()
            self._current_particle_index = self._particle_list.index(particle)
        except ValueError:
            pass  # Ignore, last particle is no longer avalable

    def _get_last_particle(self):
        """Gets the index we were at last time a visualizer of this kind was open."""
        try:
            return ParticleListVisualizer.__last_particle_by_class[type(self)]
        except KeyError:
            return -1

    def get_message_no_particles(self):
        return "No cells found. Is there some data missing?"

    def get_message_press_right(self):
        return "Press right to view the first cell."

    def draw_view(self):
        self._clear_axis()
        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            if len(self._particle_list) == 0:
                self._window.set_figure_title(self.get_message_no_particles())
            else:
                self._window.set_figure_title(self.get_message_press_right())
            self._fig.canvas.draw()
            return

        self._zoom_to_cell()
        self._show_image()

        current_particle = self._particle_list[self._current_particle_index]
        shape = self._experiment.particles.get_shape(current_particle)
        shape.draw2d(current_particle.x, current_particle.y, 0, 0, self._ax, core.COLOR_CELL_CURRENT)
        self._draw_connections(self._experiment.links, current_particle)
        self._window.set_figure_title(self.get_title(self._particle_list, self._current_particle_index))

        self._fig.canvas.draw()
        ParticleListVisualizer.__last_particle_by_class[type(self)] = current_particle

    def _zoom_to_cell(self):
        mother = self._particle_list[self._current_particle_index]
        self._ax.set_xlim(mother.x - 50, mother.x + 50)
        self._ax.set_ylim(mother.y + 50, mother.y - 50)
        self._ax.set_autoscale_on(False)

    def _draw_connections(self, links: ParticleLinks, main_particle: Particle, line_style:str = "solid",
                          line_width: int = 1):
        for connected_particle in links.find_links_of(main_particle):
            delta_time = 1
            if connected_particle.time_point_number() < main_particle.time_point_number():
                delta_time = -1
                if self._show_next_image:
                    continue  # Showing the previous position only makes things more confusing here

            color = core.COLOR_CELL_NEXT if delta_time == 1 else core.COLOR_CELL_PREVIOUS
            particle_shape = self._experiment.particles.get_shape(connected_particle)

            self._ax.plot([connected_particle.x, main_particle.x], [connected_particle.y, main_particle.y],
                          color=color, linestyle=line_style, linewidth=line_width)
            particle_shape.draw2d(connected_particle.x, connected_particle.y, 0, delta_time, self._ax, color)

    def _show_image(self):
        mother = self._particle_list[self._current_particle_index]
        time_point = self._experiment.get_time_point(mother.time_point_number())
        image_stack = self.load_image(time_point, self._show_next_image)
        if image_stack is not None:
            z = max(0, min(int(mother.z), len(image_stack) - 1))
            self._ax.imshow(image_stack[z], cmap="gray")

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

    def _exit_view(self):
        from autotrack.visualizer.standard_image_visualizer import StandardImageVisualizer

        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            # Don't know where to go
            image_visualizer = StandardImageVisualizer(self._window, display_settings=
                                                       DisplaySettings(show_next_time_point=self._show_next_image))
        else:
            mother = self._particle_list[self._current_particle_index]
            image_visualizer = StandardImageVisualizer(self._window,
                                                       time_point_number=mother.time_point_number(), z=int(mother.z),
                                                       display_settings=DisplaySettings(
                                                           show_next_time_point=self._show_next_image))
        activate(image_visualizer)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "left":
            self._goto_previous()
        elif event.key == "right":
            self._goto_next()
        elif event.key == DisplaySettings.KEY_SHOW_NEXT_IMAGE_ON_TOP:
            self._show_next_image = not self._show_next_image
            self.draw_view()
        elif event.key == "escape":
            self._exit_view()

    def _on_command(self, command: str) -> bool:
        if command == "exit":
            self._exit_view()
            return True
        if command == "help":
            self.update_status("Available commands:\n"
                               "/exit - Exits this view, and goes back to the main view.")
            return True
        return super()._on_command(command)

    def get_title(self, all_cells: List[Particle], cell_index: int):
        mother = all_cells[cell_index]
        return "Cell " + str(self._current_particle_index + 1) + "/" + str(len(self._particle_list)) + "\n" + str(mother)
