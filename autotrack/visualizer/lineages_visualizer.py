from typing import Optional, Tuple, Set, Dict, Any

from matplotlib.backend_bases import KeyEvent

from autotrack.core import TimePoint
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.linking_analysis import lineage_checks
from autotrack.visualizer import activate
from autotrack.visualizer.image_visualizer import AbstractImageVisualizer


class LineageErrorsVisualizer(AbstractImageVisualizer):
    """Viewer to detect errors in lineages. All cells with a gray marker have potential errors in them. Hover your mouse
    over a cell and press E to dismiss or correct the errors in that lineage."""

    _verified_lineages: Set[Particle]

    def __init__(self, window: Window, time_point_number: Optional[int] = None, z: int = 14):
        self._verified_lineages = set()
        super().__init__(window, time_point_number, z)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View/Exit-Exit this view (L)": self._exit_view,
        }

    def _exit_view(self, dt: int = 0):
        time_point_number = self._time_point.time_point_number() + dt
        from autotrack.visualizer.image_visualizer import StandardImageVisualizer
        image_visualizer = StandardImageVisualizer(self._window, time_point_number=time_point_number,
                                                   z=self._z, display_settings=self._display_settings)
        activate(image_visualizer)

    def _on_command(self, command: str) -> bool:
        if command == "help":
            self.update_status("Available commands:"
                               "  /t20: Jump to time point 20 (also works for other time points)"
                               "  /exit: Exits this view")
            return True
        if command == "exit":
            self._exit_view()
            return True
        return super()._on_command(command)

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
        self._exit_view(dt=dt)

    def _show_linking_errors(self, particle: Optional[Particle] = None):
        from autotrack.visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, particle)
        activate(warnings_visualizer)

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)

        # Check what lineages contain errors
        links = self._experiment.links.get_scratch_else_baseline()
        if links is None:
            self._verified_lineages = set()
            return

        particles = self._experiment.particles.of_time_point(time_point)
        lineages_with_errors = lineage_checks.get_problematic_lineages(links, particles)
        verified_lineages = set()
        for particle in particles:
            if lineage_checks.find_lineage_index_with_crumb(lineages_with_errors, particle) is None:
                verified_lineages.add(particle)
        self._verified_lineages = verified_lineages

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int) -> int:
        if dt != 0 or abs(dz) > 3:
            return super()._draw_particle(particle, color, dz, dt)

        verified = particle in self._verified_lineages
        color = color if verified else "gray"
        self._ax.plot(particle.x, particle.y, 'o', markersize=25, color=(0, 0, 0, 0), markeredgecolor=color,
                      markeredgewidth=5)
        return super()._draw_particle(particle, color, dz, dt)
