import math
from typing import Dict, Any, Optional, Tuple, List, Iterable

import numpy
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.patches import Circle

from ai_track.core import TimePoint, UserError, COLOR_CELL_CURRENT, COLOR_CELL_NEXT
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.gui.window import Window
from ai_track.imaging import lines
from ai_track.imaging.lines import Line3
from ai_track.linking import nearby_position_finder
from ai_track.linking_analysis import particle_movement_finder, particle_rotation_calculator, linking_markers
from ai_track.visualizer import activate
from ai_track.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Rotation-Rotation around an axis...": lambda: _measure_rotations(window)
    }

def _measure_rotations(window: Window):
    experiment = window.get_experiment()
    if not experiment.positions.has_positions():
        raise UserError("No positions found", "No position annotations found - cannot measure anything.")
    activate(_MeasureRotation(window))


class _Result:
    """Represents the result of the rotation calculation."""

    mean: float
    st_dev: float
    count: int
    positions: Dict[Position, List[Position]]

    def __init__(self, links: Links, resolution: ImageResolution, starting_axis_p1: Position, starting_axis_p2: Position, ending_axis_p1: Position,
                 starting_positions: Iterable[Position]):
        rotations = []
        positions = dict()

        for starting_position in starting_positions:
            if linking_markers.get_position_type(links, starting_position) == "LUMEN":
                continue  # Ignore lumens, these are not cells

            new_rotations, new_final_positions = particle_rotation_calculator.calculate_rotation_of_track(
                links, resolution, starting_position, starting_axis_p1, starting_axis_p2, ending_axis_p1)
            rotations += new_rotations
            positions[starting_position] = new_final_positions

        self.positions = positions
        self.count = len(rotations)
        if len(rotations) == 0:
            self.mean = 0
            self.st_dev = 0
        else:
            self.mean = float(numpy.mean(rotations))
            self.st_dev = float(numpy.std(rotations, ddof=1))
        print("Rotations:", rotations)


class _MeasureRotation(ExitableImageVisualizer):
    """Double-click on a point to define that as the center."""

    _axis_one_p1: Optional[Position] = None
    _axis_one_p2: Optional[Position] = None
    _radius_um: Optional[float] = None
    _axis_two_p1: Optional[Position] = None

    def _get_window_title(self) -> Optional[str]:
        return "Measuring rotation"

    def _draw_positions(self):
        if self._axis_two_p1 is not None:
            return
        super()._draw_positions()

    def _draw_links(self):
        if self._axis_two_p1 is not None:
            return
        super()._draw_links()

    def _draw_extra(self):
        # Draw first line or point
        if self._axis_one_p1 is not None and self._axis_one_p1.time_point() == self._time_point:
            if self._axis_one_p2 is None:
                self._ax.scatter(self._axis_one_p1.x, self._axis_one_p1.y, marker='X', facecolor=COLOR_CELL_CURRENT,
                                 edgecolors="black", s=17 ** 2, linewidths=2)
            else:
                self._draw_line(self._axis_one_p1, self._axis_one_p2)

        # Draw second line
        if self._axis_one_p1 is not None and self._axis_one_p2 is not None and self._axis_two_p1 is not None\
                and self._axis_two_p1.time_point() == self._time_point:
            axis_two_p2 = self._axis_two_p1 + (self._axis_one_p2 - self._axis_one_p1)
            self._draw_line(self._axis_two_p1, axis_two_p2)

    def _on_key_press(self, event: KeyEvent):
        # Find direction from key
        direction = None
        if event.key == "x":
            direction = Position(1, 0, 0)
        elif event.key == "y":
            direction = Position(0, 1, 0)
        elif event.key == "z":
            direction = Position(0, 0, 1)
        if direction is None:
            super()._on_key_press(event)
            return

        if self._axis_one_p1 is not None and self._axis_one_p2 is None:
            self._axis_one_p2 = self._axis_one_p1 + direction
            self.draw_view()
            self.update_status("Defined the rotation axis. Now click somewhere to define the radius of the cilinder.")

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick or event.xdata is None or event.ydata is None:
            return
        resolution = self._experiment.images.resolution()

        clicked_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        if self._axis_one_p1 is None or (self._axis_one_p1.time_point() != self._time_point and self._radius_um is None):
            self._axis_one_p1 = clicked_position
            self._axis_one_p2 = None
            self._radius_um = None
            self._axis_two_p1 = None
            self.draw_view()
            self.update_status("Now double-click on another point to define the rotation axis, or press X, Y or Z to"
                               " define an axis in that direction.")
        elif self._axis_one_p2 is None:
            if self._axis_one_p1.time_point() == self._time_point:
                self._axis_one_p2 = clicked_position
                self.draw_view()
                self.update_status("Defined the rotation axis. Now click somewhere to define the radius of the"
                                   " cilinder.")
            else:
                self.update_status(f"Please go back to time point {self._axis_one_p1.time_point_number()} to"
                                   f" complete the definition of the cilinder.")
        elif self._radius_um is None:
            line = Line3(self._axis_one_p1.to_vector_um(resolution), self._axis_one_p2.to_vector_um(resolution))
            self._radius_um = lines.distance_to_point(line, clicked_position.to_vector_um(resolution))
            self.draw_view()
            self.update_status(f"Defined a sphere of radius {self._radius_um:.2f} μm.\nGo to another time point to"
                               f" define the second circle center. The rotation will then be calculated.")
        elif self._axis_two_p1 is None:
            if self._time_point == self._axis_one_p1.time_point():
                self.update_status("Rotation happens over time. Go to another time point and double-click"
                                   " somewhere to calculate how much the cells have rotated.")
                return
            if self._time_point.time_point_number() < self._axis_one_p1.time_point_number():
                self.update_status("Please double-click in a time point in the future.")
                return
            self._axis_two_p1 = clicked_position
            axis_two_p2 = self._axis_two_p1 + (self._axis_one_p2 - self._axis_one_p1)
            result = self._calculate_rotation_degrees()

            # Visualize the result
            self._display_settings.z = int(self._axis_one_p1.z)
            self._move_to_time(self._axis_one_p1.time_point_number())
            for position, future_positions in result.positions.items():
                if len(future_positions) > 0:
                    self._ax.scatter(position.x, position.y, marker='o', facecolor=COLOR_CELL_CURRENT,
                                     edgecolors="black", s=17 ** 2, linewidths=2)
                for future_position in future_positions:
                    self._ax.arrow(position.x, position.y,
                                   future_position.x - position.x, future_position.y - position.y,
                                   length_includes_head=True, width=5.5, head_width=9, facecolor=COLOR_CELL_CURRENT,
                                   edgecolor="black", linewidth=2)
            self._ax.scatter(self._axis_one_p1.x, self._axis_one_p1.y, marker='X', facecolor=COLOR_CELL_NEXT,
                             edgecolors="black", s=17 ** 2, linewidths=2)
            self._ax.arrow(self._axis_one_p1.x, self._axis_one_p1.y,
                           self._axis_two_p1.x - self._axis_one_p1.x, self._axis_two_p1.y - self._axis_one_p1.y,
                           length_includes_head=True, width=5.5, head_width=9, facecolor=COLOR_CELL_NEXT,
                           edgecolor="black", linewidth=2)
            self._fig.canvas.draw()
            self.update_status(f"{result.count} cells have rotated on average {result.mean:.01f}° ± {result.st_dev:.01f}°.")

    def _calculate_rotation_degrees(self) -> _Result:
        """Returns the average rotation of the cells along with the standard deviation. The variables _axis_one,
         _axis_two and _radius_um need to be defined."""
        if self._axis_one_p1 is None or self._axis_one_p2 is None or self._axis_two_p1 is None\
                or self._radius_um is None:
            raise ValueError(f"Variables must not be None: {self._axis_one_p1}, {self._axis_one_p2}, "
                             f"{self._axis_two_p1}, {self._radius_um}")
        if self._axis_one_p1.time_point_number() > self._axis_two_p1.time_point_number():
            raise ValueError("Second time point is before first time point.")

        resolution = self._experiment.images.resolution()
        positions_in_time_point_one = self._experiment.positions.of_time_point(self._axis_one_p1.time_point())
        
        rotation_axis = Line3(self._axis_one_p1.to_vector_um(resolution), self._axis_one_p2.to_vector_um(resolution))
        positions_in_axis_one = [position for position in positions_in_time_point_one
                                   if lines.distance_to_point(rotation_axis, position.to_vector_um(resolution)) < self._radius_um]

        return _Result(self._experiment.links, resolution, self._axis_one_p1, self._axis_one_p2, self._axis_two_p1, positions_in_axis_one)

