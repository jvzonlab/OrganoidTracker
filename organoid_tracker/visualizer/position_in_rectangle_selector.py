from typing import Optional, Iterable, List

import matplotlib
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.patches import Rectangle

from organoid_tracker import core
from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor


class PositionsInRectangleSelector(LinkAndPositionEditor):
    """Define a selection by clicking somewhere, and then clicking somewhere else. All positions inside the rectangle
    defined by these two points are selected. To confirm your selection, press Enter or Escape.
    """

    _min_position: Optional[Position] = None
    _max_position: Optional[Position] = None

    _selection_rectangle: Rectangle

    def __init__(self, window: Window, *, selected_positions: Iterable[Position] = (),
                 selection_start: Optional[Position] = None):
        super().__init__(window, selected_positions=selected_positions)
        self.MAX_Z_DISTANCE = 0
        self._selection_rectangle = Rectangle((0, 0), 1, 1, fill=True,
                                              facecolor=(*matplotlib.colors.to_rgb(core.COLOR_CELL_CURRENT), 0.3),
                                              edgecolor=core.COLOR_CELL_CURRENT)
        self._min_position = selection_start

    def _get_figure_title(self) -> str:
        return ("Selecting positions, viewing time point " + str(self._time_point.time_point_number())
                + " (z=" + self._get_figure_title_z_str() + ")")

    def _exit_view(self):
        from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window, selected_positions=self._selected)
        activate(data_editor)

    def _on_mouse_move(self, event: MouseEvent):
        if (self._min_position is not None and self._max_position is None
                and event.xdata is not None and event.ydata is not None):
            # Draw a selection rectangle
            x_min = min(self._min_position.x, event.xdata)
            y_min = min(self._min_position.y, event.ydata)
            x_max = max(self._min_position.x, event.xdata)
            y_max = max(self._min_position.y, event.ydata)
            self._selection_rectangle.set_x(x_min)
            self._selection_rectangle.set_y(y_min)
            self._selection_rectangle.set_width(max(1.0, x_max - x_min))
            self._selection_rectangle.set_height(max(1.0, y_max - y_min))
            self._selection_rectangle.set_visible(True)
            self._fig.canvas.draw_idle()
        else:
            # Hide the selection rectangle
            if self._selection_rectangle.get_visible():
                self._selection_rectangle.set_visible(False)
                self._fig.canvas.draw_idle()

    def _on_mouse_single_click(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            return
        clicked_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        if (self._min_position is None and self._max_position is None) \
                or (self._min_position is not None and self._max_position is not None):
            # No positions defined yet, or both positions already defined
            self._min_position = clicked_position
            self._max_position = None
            self.get_window().redraw_data()
            return
        if self._min_position is not None and self._max_position is None:
            # One positions is defined, second is not - complete the selection
            self._set_min_max_position(self._min_position, clicked_position)
            selection_count = len(self._selected)
            self._add_positions_to_selection()
            selection_count_new = len(self._selected)
            self.get_window().redraw_data()
            self.update_status("Selected " + str(selection_count_new - selection_count) + " additional positions.\nPress"
                               " Enter or Escape to confirm the selection, or continue drawing more rectangles to"
                               " select more positions.")
            return
        # Some strange other case
        self._min_position = None
        self._max_position = None
        self.get_window().redraw_data()

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert" or event.key == "enter":
            self._exit_view()  # Confirm selection
        else:
            super()._on_key_press(event)

    def _get_newly_selected_positions(self) -> Iterable[Position]:
        """Gets all positions that are inside or outside the selected cuboid. Throws an exception if the two
        positions defining the rectangle haven't been defined yet."""
        if self._min_position is None or self._max_position is None:
            return

        if self._display_settings.max_intensity_projection:
            yield from self._get_selected_positions_2d()
            return

        for time_point_number in range(self._min_position.time_point_number(),
                                       self._max_position.time_point_number() + 1):
            time_point = TimePoint(time_point_number)
            for position in self._experiment.positions.of_time_point(time_point):
                position_is_inside = True
                if position.x < self._min_position.x or position.y < self._min_position.y \
                        or round(position.z) < self._min_position.z:
                    position_is_inside = False
                elif position.x > self._max_position.x or position.y > self._max_position.y \
                        or round(position.z) > self._max_position.z:
                    position_is_inside = False
                if position_is_inside:
                    yield position

    def _get_selected_positions_2d(self) -> Iterable[Position]:
        """Gets all positions that are inside or outside the selected rectangle, ignoring z. Throws an exception if the
        two positions defining the rectangle haven't been defined yet."""
        for time_point_number in range(self._min_position.time_point_number(),
                                       self._max_position.time_point_number() + 1):
            time_point = TimePoint(time_point_number)
            for position in self._experiment.positions.of_time_point(time_point):
                position_is_inside = True
                if position.x < self._min_position.x or position.y < self._min_position.y:
                    position_is_inside = False
                elif position.x > self._max_position.x or position.y > self._max_position.y:
                    position_is_inside = False
                if position_is_inside:
                    yield position

    def _draw_extra(self):
        # Prepare for drawing the selection rectangle
        # Invisible, but it will be made visible when needed in on_mouse_move
        self._ax.add_artist(self._selection_rectangle)

        # Draw positions already selected
        time_point_number = self._time_point.time_point_number()
        to_unselect = set()
        for selected_position in self._selected:
            is_in_z = round(selected_position.z) == self._z or self._display_settings.max_intensity_projection
            if is_in_z and selected_position.time_point_number() == time_point_number:
                if not self._experiment.positions.contains_position(selected_position):
                    to_unselect.add(selected_position)  # Position was deleted, remove from selection
                    continue

                self._draw_selection(selected_position, core.COLOR_CELL_CURRENT)

        # Unselect positions that don't exist anymore
        if len(to_unselect) > 0:
            self._selected = [element for element in self._selected if element not in to_unselect]

        # Draw selection start marker
        if self._min_position is not None and self._max_position is None:
            self._ax.scatter([self._min_position.x], [self._min_position.y], color=core.COLOR_CELL_CURRENT, s=50, marker="+")

    def _set_min_max_position(self, pos1: Position, pos2: Position):
        """Sets the minimum and maximum positions, such that the lowest x,y,z,t ends up in the lowest pos, and vice
        versa."""
        self._min_position = Position(min(pos1.x, pos2.x), min(pos1.y, pos2.y), min(pos1.z, pos2.z),
                                      time_point_number=min(pos1.time_point_number(), pos2.time_point_number()))
        self._max_position = Position(max(pos1.x, pos2.x), max(pos1.y, pos2.y), max(pos1.z, pos2.z),
                                      time_point_number=max(pos1.time_point_number(), pos2.time_point_number()))

    def _add_positions_to_selection(self):
        """Adds all positions inside the rectangle to the selection, and then deletes the selection."""
        self._selected = list(set(self._selected) | set(self._get_newly_selected_positions()))
        self._min_position = None
        self._max_position = None

