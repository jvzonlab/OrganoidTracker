import math
from typing import Dict, Any, Optional, Tuple, List

import numpy
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Circle

from ai_track.core import TimePoint, UserError, COLOR_CELL_CURRENT
from ai_track.core.position import Position
from ai_track.gui.window import Window
from ai_track.imaging import angles
from ai_track.linking import nearby_position_finder
from ai_track.linking_analysis import particle_movement_finder
from ai_track.visualizer import activate
from ai_track.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Rotation-2D rotation...": lambda: _measure_rotations(window)
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

    def __init__(self, center_one: Position, center_two: Position, positions: Dict[Position, List[Position]]):
        rotations = []

        for start_position, final_positions in positions.items():
            original_angle = angles.direction_2d(start_position, center_one)
            for future_position in final_positions:
                new_angle = angles.direction_2d(future_position, center_two)
                rotation = angles.direction_change(original_angle, new_angle)
                rotations.append(rotation)

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

    _center_one: Optional[Position] = None
    _radius_um: Optional[float] = None
    _center_two: Optional[Position] = None

    def _get_window_title(self) -> Optional[str]:
        return "Measuring rotation"

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick or event.xdata is None or event.ydata is None:
            return
        resolution = self._experiment.images.resolution()

        clicked_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        if self._center_one is None or (self._center_one.time_point() != self._time_point and self._radius_um is None):
            self._center_one = clicked_position
            self._ax.scatter([event.xdata], [event.ydata], marker='X', facecolor=COLOR_CELL_CURRENT, edgecolors="black",
                             s=17 ** 2, linewidths=2)
            self._fig.canvas.draw()
            self.update_status("Now double-click on another point to define the circle.")
        elif self._radius_um is None:
            self._radius_um = clicked_position.distance_um(self._center_one, resolution=resolution)
            radius_px = self._radius_um / resolution.pixel_size_x_um
            self._ax.add_artist(Circle((self._center_one.x, self._center_one.y), radius_px,
                                       edgecolor=COLOR_CELL_CURRENT, facecolor=(1, 1, 1, 0.2)))
            self._fig.canvas.draw()
            self.update_status(f"Defined a sphere of radius {self._radius_um:.2f} μm.\nGo to another time point to"
                               f" define the second circle center. The rotation will then be calculated.")
        else:
            if self._time_point == self._center_one.time_point():
                self.update_status("Rotation happens over time. Go to another time point and double-click"
                                   " somewhere to calculate how much the cells have rotated around the center.")
                return
            if self._time_point.time_point_number() < self._center_one.time_point_number():
                self.update_status("Please double-click in a time point in the future.")
                return
            self._center_two = clicked_position
            result = self._calculate_rotation_degrees()
            self.update_status(f"{result.count} cells have rotated on average {result.mean:.01f}° ± {result.st_dev:.01f}°.")

            # Visualize the result
            self._ax.scatter([event.xdata], [event.ydata], marker='X', facecolor=COLOR_CELL_CURRENT, edgecolors="black",
                             s=17 ** 2, linewidths=2)
            for position, future_positions in result.positions.items():
                for future_position in future_positions:
                    self._ax.arrow(position.x, position.y,
                                   future_position.x - position.x, future_position.y - position.y,
                                   length_includes_head=True, width=3, color=COLOR_CELL_CURRENT)
            self._fig.canvas.draw()

    def _calculate_rotation_degrees(self) -> _Result:
        """Returns the average rotation of the cells along with the standard deviation. The variables _center_one,
         _center_two and _radius_um need to be defined."""
        if self._center_one is None or self._center_two is None or self._radius_um is None:
            raise ValueError(f"Variables must not be None: {self._center_one}, {self._center_two}, {self._radius_um}")
        if self._center_one.time_point_number() > self._center_two.time_point_number():
            raise ValueError("Second time point is before first time point.")

        resolution = self._experiment.images.resolution()
        positions_in_time_point_one = self._experiment.positions.of_time_point(self._center_one.time_point())
        positions_in_center_one = nearby_position_finder.find_closest_n_positions(positions_in_time_point_one,
                                         around=self._center_one, max_amount=10000, resolution=resolution,
                                         max_distance_um=self._radius_um)

        links = self._experiment.links
        future_time_point = self._center_two.time_point()
        positions = dict()
        for position in positions_in_center_one:
            future_positions = particle_movement_finder.find_future_positions_at(links, position, future_time_point)
            positions[position] = future_positions

        return _Result(self._center_one, self._center_two, positions)

