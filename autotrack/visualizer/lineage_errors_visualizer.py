from typing import Optional, Set

from matplotlib.backend_bases import KeyEvent

from autotrack.core import TimePoint
from autotrack.core.particles import Particle
from autotrack.gui.window import Window
from autotrack.linking_analysis import lineage_error_finder
from autotrack.visualizer import activate
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class LineageErrorsVisualizer(ExitableImageVisualizer):
    """Viewer to detect errors in lineages. All cells with a gray marker have potential errors in them. Hover your mouse
    over a cell and press E to dismiss or correct the errors in that lineage."""

    _verified_lineages: Set[Particle] = set()

    def _on_key_press(self, event: KeyEvent):
        if event.key == "l":
            self._exit_view()
        elif event.key == "e":
            particle = self._get_particle_at(event.xdata, event.ydata)
            self._show_linking_errors(particle)
        else:
            super()._on_key_press(event)

    def _move_in_time(self, dt: int):
        # Rendering this view is quite slow, so it is better to exit this view instead of rerendering it for another
        # time point
        try:
            self._time_point = self._experiment.get_time_point(self._time_point.time_point_number() + dt)
            self._exit_view()
        except ValueError:
            pass  # Time point doesn't exit

    def _exit_view(self):
        from autotrack.visualizer.link_and_position_editor import LinkAndPositionEditor
        image_visualizer = LinkAndPositionEditor(self._window, time_point_number=self._time_point.time_point_number(),
                                                 z=self._z)
        activate(image_visualizer)

    def _show_linking_errors(self, particle: Optional[Particle] = None):
        from autotrack.visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, particle)
        activate(warnings_visualizer)

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)

        # Check what lineages contain errors
        links = self._experiment.links
        if not links.has_links():
            self._verified_lineages = set()
            return

        particles = self._experiment.particles.of_time_point(time_point)
        lineages_with_errors = lineage_error_finder.get_problematic_lineages(links, particles)
        verified_lineages = set()
        for particle in particles:
            if lineage_error_finder.find_lineage_index_with_crumb(lineages_with_errors, particle) is None:
                verified_lineages.add(particle)
        self._verified_lineages = verified_lineages

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int) -> int:
        if dt != 0 or abs(dz) > 3:
            return super()._draw_particle(particle, color, dz, dt)

        verified = particle in self._verified_lineages
        color = color if verified else "gray"
        self._draw_selection(particle, color)
        return super()._draw_particle(particle, color, dz, dt)
