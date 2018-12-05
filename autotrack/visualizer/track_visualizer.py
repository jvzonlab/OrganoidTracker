from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes

from autotrack.core import UserError
from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.core.resolution import ImageResolution
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.visualizer import DisplaySettings
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _get_particles_in_lineage(links: ParticleLinks, particle: Particle) -> ParticleLinks:
    single_lineage_links = ParticleLinks()
    _add_past_particles(links, particle, single_lineage_links)
    _add_future_particles(links, particle, single_lineage_links)
    return single_lineage_links


def _add_past_particles(links: ParticleLinks, particle: Particle, single_lineage_links: ParticleLinks):
    """Finds all particles in earlier time points connected to this particle."""
    while True:
        past_particles = links.find_pasts(particle)
        for past_particle in past_particles:
            single_lineage_links.add_link(particle, past_particle)

        if len(past_particles) == 0:
            return  # Start of lineage
        if len(past_particles) > 1:
            # Cell merge (physically impossible)
            for past_particle in past_particles:
                _add_past_particles(links, past_particle, single_lineage_links)
            return

        particle = past_particles.pop()


def _add_future_particles(links: ParticleLinks, particle: Particle, single_lineage_links: ParticleLinks):
    """Finds all particles in later time points connected to this particle."""
    while True:
        future_particles = links.find_futures(particle)
        for future_particle in future_particles:
            single_lineage_links.add_link(particle, future_particle)

        if len(future_particles) == 0:
            return  # End of lineage
        if len(future_particles) > 1:
            # Cell division
            for daughter in future_particles:
                _add_future_particles(links, daughter, single_lineage_links)
            return

        particle = future_particles.pop()


def _plot_displacements(axes: Axes, links: ParticleLinks, resolution: ImageResolution, particle: Particle):
    displacements = list()
    time_point_numbers = list()

    while True:
        future_particles = links.find_futures(particle)

        if len(future_particles) == 1:
            # Track continues
            future_particle = future_particles.pop()
            delta_time = future_particle.time_point_number() - particle.time_point_number()
            displacements.append(particle.distance_um(future_particle, resolution) / delta_time)
            time_point_numbers.append(particle.time_point_number())

            particle = future_particle
            continue

        # End of this cell track: either start multiple new ones (division) or stop tracking
        axes.plot(time_point_numbers, displacements)
        for future_particle in future_particles:
            _plot_displacements(axes, links, resolution, future_particle)
        return


class TrackVisualizer(ExitableImageVisualizer):
    """Shows the trajectory of a single cell. Double-click a cell to select it. Press T to exit this view."""

    _particles_in_lineage: Optional[ParticleLinks] = None

    def __init__(self, window: Window, time_point_number: int,
                 z: int, display_settings: DisplaySettings):
        super().__init__(window, time_point_number=time_point_number, z=z, display_settings=display_settings)

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int) -> int:
        if abs(dz) <= 3 and self._particles_in_lineage is not None\
                and self._particles_in_lineage.contains_particle(particle):
            self._draw_selection(particle, color)
        return super()._draw_particle(particle, color, dz, dt)

    def _get_figure_title(self):
        return f"Tracks at time point {self._time_point.time_point_number()} (z={self._z})"

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
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
                for lineage_start in self._particles_in_lineage.find_appeared_cells():
                    _plot_displacements(axes, self._particles_in_lineage, resolution, lineage_start)

            dialog.popup_figure(self.get_window().get_gui_experiment(), draw_function)

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
            links = self._experiment.links
            if not links.has_links():
                self.update_status("No links found. Is the linking data missing?")
                return
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is None:
                self.update_status("Couldn't find a particle here.")
                self._particles_in_lineage = None
                return
            self._particles_in_lineage = _get_particles_in_lineage(links, particle)
            self.draw_view()
            self.update_status("Focused on " + str(particle))
