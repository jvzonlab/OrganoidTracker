from typing import Optional, Dict

import matplotlib

from autotrack.core import TimePoint
from autotrack.core.position import Position
from autotrack.gui.window import Window
from autotrack.position_analysis import cell_curvature_calculator
from autotrack.visualizer import DisplaySettings
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class CellCurvatureVisualizer(ExitableImageVisualizer):
    """Shows the curvature around a cell: the average angle of any nearby cell to an opposite cell via the original
     cell."""

    _min_cell_curvature: Optional[float] = None
    _max_cell_curvature: Optional[float] = None
    _position_to_curvature: Dict[Position, float]

    def __init__(self, window: Window, *, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)

        self._position_to_curvature = dict()
        self._calculate_densities()

    def refresh_data(self):
        self._calculate_densities()
        super().refresh_data()

    def refresh_all(self):
        self._calculate_densities()
        super().refresh_all()

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)
        self._calculate_densities()

    def _calculate_densities(self):
        experiment = self._experiment
        positions = self._experiment.positions.of_time_point(self._time_point)
        min_curvature = None
        max_curvature = None
        densities = dict()

        for position in positions:
            cell_curvature = cell_curvature_calculator.get_curvature_angle(experiment, position)
            if min_curvature is None or cell_curvature < min_curvature:
                min_curvature = cell_curvature
            if max_curvature is None or cell_curvature > max_curvature:
                max_curvature = cell_curvature
            densities[position] = cell_curvature

        self._min_cell_curvature = min_curvature
        self._max_cell_curvature = max_curvature
        self._position_to_curvature = densities

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        color_map = matplotlib.cm.get_cmap("jet")

        curvature = self._position_to_curvature.get(position)
        curvature_color = "gray"
        if curvature is not None:
            curvature_color = color_map((curvature - self._min_cell_curvature)/(self._max_cell_curvature - self._min_cell_curvature))
        if dt == 0 and abs(dz) <= 3:
            self._draw_selection(position, curvature_color)
