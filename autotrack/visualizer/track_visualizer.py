from typing import Optional, List

import matplotlib.cm
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes

from autotrack.core import UserError, TimePoint
from autotrack.core.links import Links, LinkingTrack
from autotrack.core.positions import Position
from autotrack.core.resolution import ImageResolution
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.visualizer import DisplaySettings
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _get_lineage(links: Links, position: Position) -> Links:
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


def _plot_displacements(axes: Axes, resolution: ImageResolution, track: LinkingTrack, lineage_id: int):
    displacements = list()
    time_point_numbers = list()

    previous_position = None
    for position in track.positions():
        if previous_position is not None:
            # Found a connection, measure distance
            delta_time = position.time_point_number() - previous_position.time_point_number()
            displacements.append(position.distance_um(previous_position, resolution) / delta_time)
            time_point_numbers.append(position.time_point_number())

        previous_position = position

    # End of this cell track: either start multiple new ones (division) or stop tracking
    label = None if track.get_previous_tracks() else "Lineage " + str(lineage_id)
    axes.plot(time_point_numbers, displacements, color=matplotlib.cm.Set1(lineage_id), label=label)


class TrackVisualizer(ExitableImageVisualizer):
    """Shows the trajectory of one or multiple particles. Double-click a particle to select it. Double click it again to
    deselect it. Double-click on an empty area to deselect all particles."""

    _selected_lineages: List[Links]

    def __init__(self, window: Window, time_point: TimePoint, z: int, display_settings: DisplaySettings):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)
        self._selected_lineages = list()

    def _draw_position(self, position: Position, color: str, dz: int, dt: int):
        if abs(dz) > 3 or len(self._selected_lineages) == 0:
            # Draw normally
            super()._draw_position(position, color, dz, dt)
            return

        for i, lineage in enumerate(self._selected_lineages):
            if not lineage.contains_position(position):
                continue  # Part of another lineage

            # Draw marker or a number
            if dt == 0 and len(self._selected_lineages) > 1:
                # There are multiple tracks show
                self._draw_annotation(position, str(i + 1))
            else:
                super()._draw_position(position, color, dz, dt)  # Draw normally

            # Mark as selected
            self._draw_selection(position, color)
            return

        # Didn't draw any selection marker, so draw normally
        super()._draw_position(position, color, dz, dt)

    def _get_figure_title(self):
        return f"Tracks at time point {self._time_point.time_point_number()} (z={self._z})"

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Graph//Over time-Cell displacement over time...": self._show_displacement,
        }

    def _show_displacement(self):
        resolution = self._experiment.image_resolution()

        if len(self._selected_lineages) == 0:
            raise UserError("No cell track selected", "No cell track selected, so we cannot plot anything. Double-click"
                                                      " on a cell to select a track.")

        def draw_function(figure: Figure):
            axes = figure.gca()
            axes.set_xlabel("Time (time points)")
            axes.set_ylabel("Displacement between time points (μm)")
            axes.set_title("Cellular displacement")
            for i, lineage in enumerate(self._selected_lineages):
                for track in lineage.find_all_tracks():
                    _plot_displacements(axes, resolution, track, i + 1)
            if len(self._selected_lineages) > 1:
                axes.legend()

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
        if not event.dblclick:
            return

        links = self._experiment.links
        if not links.has_links():
            self.update_status("No links found. Is the linking data missing?")
            return

        # Get the clicked cell
        position = self._get_position_at(event.xdata, event.ydata)
        if position is None:
            # Deselect everything if we didn't click on a cell
            self._selected_lineages.clear()
            self.draw_view()
            self.update_status("Couldn't find a position here, deselecting all lineages.")
            return

        for i in range(len(self._selected_lineages)):
            lineage = self._selected_lineages[i]
            if lineage.contains_position(position):
                del self._selected_lineages[i]  # Deselect lineage
                self.draw_view()
                self.update_status("Deselected " + str(position))
                return

        self._selected_lineages.append(_get_lineage(links, position))
        self.draw_view()
        self.update_status("Selected " + str(position))

