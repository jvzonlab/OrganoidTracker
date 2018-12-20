from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes

from autotrack.core import UserError, TimePoint
from autotrack.core.links import Links
from autotrack.core.positions import Position
from autotrack.core.resolution import ImageResolution
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.visualizer import DisplaySettings
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _get_positions_in_lineage(links: Links, position: Position) -> Links:
    single_lineage_links = Links()
    _add_past_positions(links, position, single_lineage_links)
    _add_future_positions(links, position, single_lineage_links)
    return single_lineage_links


def _add_past_positions(links: Links, position: Position, single_lineage_links: Links):
    """Finds all positions in earlier time points connected to this position."""
    while True:
        past_positions = links.find_pasts(position)
        for past_position in past_positions:
            single_lineage_links.add_link(position, past_position)

        if len(past_positions) == 0:
            return  # Start of lineage
        if len(past_positions) > 1:
            # Cell merge (physically impossible)
            for past_position in past_positions:
                _add_past_positions(links, past_position, single_lineage_links)
            return

        position = past_positions.pop()


def _add_future_positions(links: Links, position: Position, single_lineage_links: Links):
    """Finds all positions in later time points connected to this position."""
    while True:
        future_positions = links.find_futures(position)
        for future_position in future_positions:
            single_lineage_links.add_link(position, future_position)

        if len(future_positions) == 0:
            return  # End of lineage
        if len(future_positions) > 1:
            # Cell division
            for daughter in future_positions:
                _add_future_positions(links, daughter, single_lineage_links)
            return

        position = future_positions.pop()


def _plot_displacements(axes: Axes, links: Links, resolution: ImageResolution, position: Position):
    displacements = list()
    time_point_numbers = list()

    while True:
        future_positions = links.find_futures(position)

        if len(future_positions) == 1:
            # Track continues
            future_position = future_positions.pop()
            delta_time = future_position.time_point_number() - position.time_point_number()
            displacements.append(position.distance_um(future_position, resolution) / delta_time)
            time_point_numbers.append(position.time_point_number())

            position = future_position
            continue

        # End of this cell track: either start multiple new ones (division) or stop tracking
        axes.plot(time_point_numbers, displacements)
        for future_position in future_positions:
            _plot_displacements(axes, links, resolution, future_position)
        return


class TrackVisualizer(ExitableImageVisualizer):
    """Shows the trajectory of a single cell. Double-click a cell to select it. Press T to exit this view."""

    _positions_in_lineage: Optional[Links] = None

    def __init__(self, window: Window, time_point: TimePoint, z: int, display_settings: DisplaySettings):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)

    def _draw_position(self, position: Position, color: str, dz: int, dt: int) -> int:
        if abs(dz) <= 3 and self._positions_in_lineage is not None\
                and self._positions_in_lineage.contains_position(position):
            self._draw_selection(position, color)
        return super()._draw_position(position, color, dz, dt)

    def _get_figure_title(self):
        return f"Tracks at time point {self._time_point.time_point_number()} (z={self._z})"

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Graph//Over time-Cell displacement over time...": self._show_displacement,
        }

    def _show_displacement(self):
        resolution = self._experiment.image_resolution()

        if self._positions_in_lineage is None:
            raise UserError("No cell track selected", "No cell track selected, so we cannot plot anything. Double-click"
                                                      " on a cell to select a track.")

        def draw_function(figure: Figure):
            axes = figure.gca()
            axes.set_xlabel("Time (time points)")
            axes.set_ylabel("Displacement between time points (μm)")
            axes.set_title("Cellular displacement")
            for lineage_start in self._positions_in_lineage.find_appeared_cells():
                _plot_displacements(axes, self._positions_in_lineage, resolution, lineage_start)

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
            position = self._get_position_at(event.xdata, event.ydata)
            if position is None:
                self.update_status("Couldn't find a position here.")
                self._positions_in_lineage = None
                return
            self._positions_in_lineage = _get_positions_in_lineage(links, position)
            self.draw_view()
            self.update_status("Focused on " + str(position))
