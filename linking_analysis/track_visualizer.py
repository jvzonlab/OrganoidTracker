from imaging.visualizer import Visualizer, activate
from imaging import Experiment, Particle, cell
from matplotlib.figure import Figure
from matplotlib.backend_bases import KeyEvent, MouseEvent
from networkx import Graph
from typing import Set, Optional
import matplotlib.pyplot as plt


class TrackVisualizer(Visualizer):
    """Shows trajectories of particles. The past is brown, the future is orange.
    Double-click on a cell point to focus on that cell.
    Press T to return to the normal view."""

    _particle: Particle
    _particles_on_display: Set[Particle]

    def __init__(self, experiment: Experiment, figure: Figure, particle: Particle):
        super().__init__(experiment, figure)
        self._particle = particle
        self._particles_on_display = set()

    def draw_view(self):
        self._clear_axis()
        self._particles_on_display.clear()

        self._draw_particle(self._particle, color='purple', size=7)

        self._draw_network(self._experiment.particle_links_scratch(), line_style='dotted', line_width=3, max_distance=1)
        self._draw_network(self._experiment.particle_links())

        plt.title("Tracks of particle " + str(self._particle) + "\n" + self._get_cell_age_str())
        plt.draw()

    def _get_cell_age_str(self) -> str:
        graph = self._experiment.particle_links()
        if graph is None:
            return ""
        age = cell.get_age(self._experiment, graph, self._particle)
        if age is None:
            return "Age: born before measurements"
        return "Age: " + str(age)

    def _draw_network(self, network: Optional[Graph], line_style: str = 'solid', line_width: int = 1,
                      max_distance: int = 10):
        if network is not None:
            already_drawn = {self._particle}
            self._draw_all_connected(self._particle, network, already_drawn, line_style=line_style,
                                     line_width=line_width, max_distance=max_distance)
            self._particles_on_display.update(already_drawn)

    def _draw_all_connected(self, particle: Particle, network: Graph, already_drawn: Set[Particle],
                            max_distance: int = 10, line_style: str = 'solid', line_width: float = 1):
        try:
            links = network[particle]
            for linked_particle in links:
                if linked_particle not in already_drawn:
                    color = 'orange'
                    positions = [particle.x, particle.y, linked_particle.x - particle.x, linked_particle.y - particle.y]
                    if linked_particle.frame_number() <= self._particle.frame_number():
                        # Particle in the past, use different style
                        color = 'darkred'
                    if linked_particle.frame_number() < particle.frame_number():
                        # Always draw arrow from oldest to newest particle
                        positions = [linked_particle.x, linked_particle.y,
                                     particle.x - linked_particle.x, particle.y - linked_particle.y]

                    self._ax.arrow(*positions, color=color, linestyle=line_style, linewidth=line_width, fc=color,
                                   ec=color, head_width=1, head_length=1)
                    if abs(linked_particle.frame_number() - self._particle.frame_number()) <= max_distance:
                        already_drawn.add(linked_particle)
                        self._draw_particle(linked_particle, color)
                        self._draw_all_connected(linked_particle, network, already_drawn, max_distance, line_style,
                                                 line_width)
        except KeyError:
            pass # No older links

    def _draw_particle(self, particle: Particle, color, size=5):
        self._ax.plot(particle.x, particle.y, 'o', color=color, markeredgecolor='black', markersize=size)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            from imaging.image_visualizer import StandardImageVisualizer
            image_visualizer = StandardImageVisualizer(self._experiment, self._fig, z=int(self._particle.z),
                                                       frame_number=self._particle.frame_number())
            activate(image_visualizer)

    def _on_mouse_click(self, event: MouseEvent):
        if event.button == 1 and event.dblclick:
            # Focus on clicked particle
            particle = self.get_closest_particle(self._particles_on_display, x=event.xdata, y=event.ydata, z=None,
                                                 max_distance=20)
            if particle is not None:
                self._particle = particle
                self.draw_view()
