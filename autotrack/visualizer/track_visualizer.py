from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes
from networkx import Graph

from autotrack.core import UserError
from autotrack.core.particles import Particle
from autotrack.core.resolution import ImageResolution
from autotrack.gui import Window, dialog
from autotrack.linking import existing_connections
from autotrack.linking_analysis import cell_appearance_finder
from autotrack.visualizer import DisplaySettings
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _get_particles_in_lineage(graph: Graph, particle: Particle) -> Graph:
    particles = Graph()
    particles.add_node(particle)
    _add_past_particles(graph, particle, particles)
    _add_future_particles(graph, particle, particles)
    return particles


def _add_past_particles(graph: Graph, particle: Particle, single_lineage_graph: Graph):
    """Finds all particles in earlier time points connected to this particle."""
    while True:
        past_particles = existing_connections.find_past_particles(graph, particle)
        for past_particle in past_particles:
            single_lineage_graph.add_node(past_particle)
            single_lineage_graph.add_edge(particle, past_particle)

        if len(past_particles) == 0:
            return  # Start of lineage
        if len(past_particles) > 1:
            # Cell merge (physically impossible)
            for past_particle in past_particles:
                _add_past_particles(graph, past_particle, single_lineage_graph)
            return

        particle = past_particles.pop()


def _add_future_particles(graph: Graph, particle: Particle, single_lineage_graph: Graph):
    """Finds all particles in later time points connected to this particle."""
    while True:
        future_particles = existing_connections.find_future_particles(graph, particle)
        for future_particle in future_particles:
            single_lineage_graph.add_node(future_particle)
            single_lineage_graph.add_edge(particle, future_particle)

        if len(future_particles) == 0:
            return  # End of lineage
        if len(future_particles) > 1:
            # Cell division
            for daughter in future_particles:
                _add_future_particles(graph, daughter, single_lineage_graph)
            return

        particle = future_particles.pop()


def _plot_displacements(axes: Axes, graph: Graph, resolution: ImageResolution, particle: Particle):
    displacements = list()
    time_point_numbers = list()

    while True:
        future_particles = existing_connections.find_future_particles(graph, particle)

        if len(future_particles) == 1:
            # Track continues
            future_particle = future_particles.pop()
            displacements.append(particle.distance_um(future_particle, resolution))
            time_point_numbers.append(particle.time_point_number())

            particle = future_particle
            continue

        # End of this cell track: either start multiple new ones (division) or stop tracking
        axes.plot(time_point_numbers, displacements)
        for future_particle in future_particles:
            _plot_displacements(axes, graph, resolution, future_particle)
        return


class TrackVisualizer(ExitableImageVisualizer):
    """Shows the trajectory of a single cell. Double-click a cell to select it. Press T to exit this view."""

    _particles_in_lineage: Optional[Graph] = None

    def __init__(self, window: Window, time_point_number: int,
                 z: int, display_settings: DisplaySettings):
        super().__init__(window, time_point_number, z, display_settings)

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int) -> int:
        if abs(dz) <= 3 and self._particles_in_lineage is not None and particle in self._particles_in_lineage:
            self._draw_selection(particle, color)
        return super()._draw_particle(particle, color, dz, dt)

    def _get_figure_title(self):
        return f"Tracks at time point {self._time_point.time_point_number()} (z={self._z})"

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "View/Exit-Exit this view (T)": self._exit_view,
            "Graph/Displacement-Cell displacement over time...": self._show_displacement,
        }

    def _show_displacement(self):
        try:
            resolution = self._experiment.image_resolution()
        except ValueError:
            raise UserError("Resolution not set", "The image resolution is not set. Cannot calculate cellular"
                                                  " displacement")
        else:
            if self._particles_in_lineage is None:
                raise UserError("No cell track selected", "No cell track selected, so we cannot plot anything. Double-click"
                                                          " on a cell to select a track.")

            def draw_function(figure: Figure):
                axes = figure.gca()
                axes.set_xlabel("Time (time points)")
                axes.set_ylabel("Displacement between time points (μm)")
                axes.set_title("Cellular displacement")
                for lineage_start in cell_appearance_finder.find_appeared_cells(self._particles_in_lineage):
                    _plot_displacements(axes, self._particles_in_lineage, resolution, lineage_start)

            dialog.popup_figure(self._experiment.name, draw_function)

    def _on_command(self, command: str) -> bool:
        if command == "exit":
            self._exit_view()
            return True
        return super()._on_command(command)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            self._exit_view()
        else:
            super()._on_key_press(event)

    def _on_mouse_click(self, event: MouseEvent):
        if event.dblclick:
            graph = self._experiment.links.graph
            if graph is None:
                self.update_status("No links found. Is the linking data missing?")
                return
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is None:
                self.update_status("Couldn't find a particle here.")
                self._particles_in_lineage = None
                return
            self._particles_in_lineage = _get_particles_in_lineage(graph, particle)
            self.draw_view()
            self.update_status("Focused on " + str(particle))
