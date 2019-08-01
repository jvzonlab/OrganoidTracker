from typing import Optional, Iterable, List

from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.patches import Rectangle

from ai_track import core
from ai_track.core import TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.particle import Particle
from ai_track.core.position import Position
from ai_track.gui.undo_redo import UndoableAction
from ai_track.linking_analysis import cell_error_finder
from ai_track.visualizer import activate
from ai_track.visualizer.abstract_editor import AbstractEditor


class _DeletePositionsAction(UndoableAction):

    _particles: List[Particle]

    def __init__(self, particles: Iterable[Particle]):
        self._particles = list(particles)

    def do(self, experiment: Experiment):
        experiment.remove_positions((particle.position for particle in self._particles))
        for particle in self._particles:  # Check linked particles for errors
            cell_error_finder.find_errors_in_positions_and_all_dividing_cells(experiment, *particle.links)
        return f"Removed all {len(self._particles)} positions within the rectangle"

    def undo(self, experiment: Experiment):
        for particle in self._particles:
            particle.restore(experiment)
            cell_error_finder.find_errors_in_positions_and_all_dividing_cells(experiment, particle.position, *particle.links)
        return f"Re-added {len(self._particles)} positions"


class PositionsInRectangleDeleter(AbstractEditor):
    """Double click to define the first point, double-click again to define the second point. Then press Delete to
    delete all positions within that rectangle."""

    _min_position: Optional[Position] = None
    _max_position: Optional[Position] = None

    def _get_figure_title(self) -> str:
        return "Deleting positions, viewing time point " \
               + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _exit_view(self):
        from ai_track.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window, time_point=self._time_point, z=self._z,
                                            display_settings=self._display_settings)
        activate(data_editor)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "delete":
            self._try_delete()
            return
        super()._on_key_press(event)

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return

        clicked_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        if self._min_position is None:
            # No positions defined yet
            self._min_position = clicked_position
            self.get_window().redraw_data()
            return
        if self._max_position is None:
            # One positions is defined
            self._set_min_max_position(self._min_position, clicked_position)
            self.get_window().redraw_data()
            width = self._max_position.x - self._min_position.x + 1
            height = self._max_position.y - self._min_position.y + 1
            depth = self._max_position.z - self._min_position.z + 1
            time = self._max_position.time_point_number() - self._min_position.time_point_number() + 1
            self.update_status(f"Selected a volume of {width}x{height}x{depth} px, spanning {time} time points")
            return
        # Both positions are defined
        self._min_position = None
        self._max_position = None
        self.get_window().redraw_data()

    def _is_rectangle_at_current_time(self) -> bool:
        """Returns True if the rectangle overlaps with this time point and z value."""
        if self._min_position.time_point_number() > self._time_point.time_point_number():
            return False
        if self._max_position.time_point_number() < self._time_point.time_point_number():
            return False
        return True

    def _is_rectangle_at_current_layer(self) -> bool:
        if self._min_position.z > self._z:
            return False
        if self._max_position.z < self._z:
            return False
        return True

    def _get_positions_in_rectangle(self) -> Iterable[Position]:
        """Gets all positions that are within the selected rectangle. Throws an exception if the two positions defining
        the rectangle haven't been defined yet."""
        for time_point_number in range(self._min_position.time_point_number(), self._max_position.time_point_number() + 1):
            time_point = TimePoint(time_point_number)
            for position in self._experiment.positions.of_time_point(time_point):
                if position.x < self._min_position.x or position.y < self._min_position.y\
                        or position.z < self._min_position.z:
                    continue
                if position.x > self._max_position.x or position.y > self._max_position.y\
                        or position.z > self._max_position.z:
                    continue
                yield position

    def _draw_extra(self):
        if self._max_position is not None and self._min_position is not None:
            # We can draw a rectangle
            is_at_t = self._is_rectangle_at_current_time()
            is_at_z = self._is_rectangle_at_current_layer()

            width = self._max_position.x - self._min_position.x
            height = self._max_position.y - self._min_position.y
            facecolor = "white" if is_at_t and is_at_z else None
            alpha = 0.5 if is_at_t and is_at_z else None
            edgecolor = core.COLOR_CELL_CURRENT
            fill = is_at_z and is_at_t
            if not is_at_t:
                edgecolor = core.COLOR_CELL_PREVIOUS \
                    if self._time_point.time_point_number() > self._max_position.time_point_number() \
                    else core.COLOR_CELL_NEXT
            rectangle = Rectangle(xy=(self._min_position.x, self._min_position.y), width=width, height=height,
                                  fill=fill, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
            self._ax.add_artist(rectangle)
        if self._min_position is not None and self._max_position is None:
            self._draw_selection(self._min_position, core.COLOR_CELL_CURRENT)

    def _set_min_max_position(self, pos1: Position, pos2: Position):
        """Sets the minimum and maximum positions, such that the lowest x,y,z,t ends up in the lowest pos, and vice
        versa."""
        self._min_position = Position(min(pos1.x, pos2.x), min(pos1.y, pos2.y), min(pos1.z, pos2.z),
                                      time_point_number=min(pos1.time_point_number(), pos2.time_point_number()))
        self._max_position = Position(max(pos1.x, pos2.x), max(pos1.y, pos2.y), max(pos1.z, pos2.z),
                                      time_point_number=max(pos1.time_point_number(), pos2.time_point_number()))

    def _try_delete(self):
        if self._min_position is None or self._max_position is None:
            self.update_status("Please select a rectangle first. Double-click somewhere to define the corners.")
            return

        experiment = self._experiment
        positions = [Particle.from_position(experiment, position)
                                                     for position in self._get_positions_in_rectangle()]
        if len(positions) == 0:
            self.update_status("There are no positions within the selected rectangle")
            return
        self._min_position = None
        self._max_position = None
        self._perform_action(_DeletePositionsAction(positions))
