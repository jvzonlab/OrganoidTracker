from typing import Optional, List, Set

import matplotlib.cm
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes

from organoid_tracker.coordinate_system import orientation_spline_adder, sphere_representer
from organoid_tracker.coordinate_system.orientation_spline_adder import ColoredTrackAdder
from organoid_tracker.coordinate_system.sphere_representer import SphereRepresentation
from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.lineage_drawing import LineageDrawing
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


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


def _plot_volume(axes: Axes, position_data: PositionData, track: LinkingTrack, lineage_id: int):
    volumes = list()
    time_point_numbers = list()

    for position in track.positions():
        # Found a connection, measure distance
        try:
            volumes.append(linking_markers.get_shape(position_data, position).volume())
            time_point_numbers.append(position.time_point_number())
        except NotImplementedError:
            pass  # No known volume at this time point

    # End of this cell track: either start multiple new ones (division) or stop tracking
    label = None if track.get_previous_tracks() else "Lineage " + str(lineage_id)
    axes.plot(time_point_numbers, volumes, color=matplotlib.cm.Set1(lineage_id), label=label)


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


def _plot_data_axes_locations(axes: Axes, experiment: Experiment, track: LinkingTrack, lineage_id: int):
    data_axes = experiment.splines
    links = experiment.links
    resolution = experiment.images.resolution()

    data_axis_locations_um = list()
    time_point_numbers = list()

    # Connect line with previous track
    positions = list(track.positions())
    previous_tracks = track.get_previous_tracks()
    if len(previous_tracks) == 1:
        positions.insert(0, previous_tracks.pop().find_last_position())

    # Plot current track
    for position in positions:
        data_axis_location = data_axes.to_position_on_original_axis(links, position)
        if data_axis_location is None:
            continue

        data_axis_locations_um.append(data_axis_location.pos * resolution.pixel_size_x_um)
        time_point_numbers.append(position.time_point_number())

    if len(data_axis_locations_um) == 0:
        raise UserError("No data axes", "No data axes found. Did you forget to draw a data axis for the positions of"
                                        "this track?")

    # End of this cell track: either start multiple new ones (division) or stop tracking
    label = None if track.get_previous_tracks() else "Lineage " + str(lineage_id)
    axes.plot(time_point_numbers, data_axis_locations_um, color=matplotlib.cm.Set1(lineage_id), label=label)


class TrackVisualizer(ExitableImageVisualizer):
    """Shows the trajectory of one or multiple particles. Double-click a particle to select it. Double click it again to
    deselect it. Double-click on an empty area to deselect all particles."""

    _selected_lineages: List[Links]

    def __init__(self, window: Window):
        super().__init__(window)
        self._selected_lineages = list()

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if abs(dz) > 3 or len(self._selected_lineages) == 0:
            # Draw normally
            super()._on_position_draw(position, color, dz, dt)
            return True

        for i, lineage in enumerate(self._selected_lineages):
            if not lineage.contains_position(position):
                continue  # Part of another lineage

            # Draw marker or a number
            if dt == 0 and len(self._selected_lineages) > 1:
                # There are multiple tracks show
                self._draw_annotation(position, str(i + 1))
            else:
                super()._on_position_draw(position, color, dz, dt)  # Draw normally

            # Mark as selected
            self._draw_selection(position, color)
            return True

        # Didn't draw any selection marker, so draw normally
        return super()._on_position_draw(position, color, dz, dt)

    def _get_figure_title(self):
        return f"Tracks at time point {self._time_point.time_point_number()} (z={self._z})"

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Graph//Over time-Displacement over time...": self._show_displacement,
            "Graph//Over time-Axes positions over time...": self._show_data_axes_locations,
            "Graph//Over time-Volume over time...": self._show_volumes,
            "Graph//Visualization-Lineage tree...": self._show_lineage_tree,
            "Graph//Visualization-Projection on sphere...": self._show_sphere,
            "Graph//Visualization-Projection on world map...": self._show_world_map
        }

    def _show_displacement(self):
        resolution = self._experiment.images.resolution()

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

    def _show_data_axes_locations(self):
        if len(self._selected_lineages) == 0:
            raise UserError("No cell track selected", "No cell track selected, so we cannot plot anything. Double-click"
                                                      " on a cell to select a track.")

        def draw_function(figure: Figure):
            axes = figure.gca()
            axes.set_xlabel("Time (time points)")
            axes.set_ylabel("Position on axis (μm)")
            axes.set_title("Movement of cells on axis")
            for i, lineage in enumerate(self._selected_lineages):
                for track in lineage.find_all_tracks():
                    _plot_data_axes_locations(axes, self._experiment, track, i + 1)
            if len(self._selected_lineages) > 1:
                axes.legend()

        dialog.popup_figure(self.get_window().get_gui_experiment(), draw_function)

    def _show_volumes(self):
        if len(self._selected_lineages) == 0:
            raise UserError("No cell track selected", "No cell track selected, so we cannot plot anything. Double-click"
                                                      " on a cell to select a track.")

        def draw_function(figure: Figure):
            axes = figure.gca()
            axes.set_xlabel("Time (time points)")
            axes.set_ylabel("Volume (px$^3$)")
            axes.set_title("Volume of the cells")
            for i, lineage in enumerate(self._selected_lineages):
                for track in lineage.find_all_tracks():
                    _plot_volume(axes, self._experiment.position_data, track, i + 1)
            if len(self._selected_lineages) > 1:
                axes.legend()

        dialog.popup_figure(self.get_window().get_gui_experiment(), draw_function)

    def _show_lineage_tree(self):
        lineage_drawing = LineageDrawing(self._experiment.links)
        tracks = self._find_lineage_tracks()

        def lineage_filter(track: LinkingTrack) -> bool:
            return track in tracks

        def color_getter(time_point_number: int, track: LinkingTrack) -> MPLColor:
            position = track.find_position_at_time_point_number(time_point_number)
            for i, lineage in enumerate(self._selected_lineages):
                if lineage.contains_position(position):
                    return tuple(SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)])
            return "black"

        def label_getter(track: LinkingTrack) -> Optional[str]:
            if len(track.get_previous_tracks()) != 0:
                return None  # Not a start of a lineage, don't label
            position = track.find_first_position()
            for i, lineage in enumerate(self._selected_lineages):
                if lineage.contains_position(position):
                    return str(i + 1)
            return None

        def draw_function(figure: Figure):
            experiment = self._experiment
            axes: Axes = figure.gca()
            width = lineage_drawing.draw_lineages_colored(axes, color_getter=color_getter,
                                                          lineage_filter=lineage_filter,
                                                          label_getter=label_getter)
            axes.set_xlim(-1, width + 1)
            axes.set_ylim([experiment.last_time_point_number(), experiment.first_time_point_number() - 1])
            axes.set_ylabel("Time point")
            axes.set_xticks([])

        dialog.popup_figure(self.get_window().get_gui_experiment(), draw_function)

    def _show_sphere(self):
        experiment = self._experiment
        sphere_representation = SphereRepresentation(experiment.beacons, 1, experiment.images.resolution())
        orientation_spline_adder.add_all_splines(sphere_representation, experiment.splines)
        track_adder = ColoredTrackAdder(experiment)
        for lineage in self._selected_lineages:
            for track in lineage.find_all_tracks():
                track_adder.add_track_colored_by_spline_position(sphere_representation, track.positions())

        def draw_function(figure: Figure):
            sphere_representer.setup_figure_3d(figure, sphere_representation)

        dialog.popup_figure(self.get_window().get_gui_experiment(), draw_function)

    def _show_world_map(self):
        experiment = self._experiment
        sphere_representation = SphereRepresentation(experiment.beacons, 1, experiment.images.resolution())
        orientation_spline_adder.add_all_splines(sphere_representation, experiment.splines)
        track_adder = ColoredTrackAdder(experiment)
        for lineage in self._selected_lineages:
            for track in lineage.find_all_tracks():
                track_adder.add_track_colored_by_spline_position(sphere_representation, track.positions())

        def draw_function(figure: Figure):
            ax = figure.gca()
            sphere_representer.setup_figure_2d(ax, sphere_representation)

        dialog.popup_figure(self.get_window().get_gui_experiment(), draw_function)

    def _find_lineage_tracks(self) -> Set[LinkingTrack]:
        """Finds all linking tracks in the same lineage, including sisters, daughters, etc."""
        links = self._experiment.links
        result = set()

        for lineage in self._selected_lineages:
            for starting_track in lineage.find_starting_tracks():
                starting_position = starting_track.find_first_position()
                track = links.get_track(starting_position)
                if track is not None:
                    result |= set(track.find_all_descending_tracks(include_self=True))
        return result

    def _is_selected(self, position: Position) -> bool:
        """Checks if the given position is currently part of a selected track."""
        for i, lineage in enumerate(self._selected_lineages):
            if lineage.contains_position(position):
                return True
        return False

    def _move_to_position(self, position: Position) -> bool:
        # Select that lineage
        if not self._is_selected(position):
            self._selected_lineages.append(_get_lineage(self._experiment.links, position))
        return super()._move_to_position(position)

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

