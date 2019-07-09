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

    _min_cell_curvature: float = 0
    _max_cell_curvature: float = 50
    _position_to_curvature: Dict[Position, float]

    def __init__(self, window: Window, *, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)

        self._position_to_curvature = dict()
        self._calculate_curvatures()

    def refresh_data(self):
        self._calculate_curvatures()
        super().refresh_data()

    def refresh_all(self):
        self._calculate_curvatures()
        super().refresh_all()

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)
        self._calculate_curvatures()

    def _calculate_curvatures(self):
        experiment = self._experiment
        positions = self._experiment.positions.of_time_point(self._time_point)
        curvatures = dict()

        for position in positions:
            cell_curvature = cell_curvature_calculator.get_curvature_angle(experiment.positions, position,
                                                                           experiment.images.resolution())
            curvatures[position] = cell_curvature

        self._position_to_curvature = curvatures

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        color_map = matplotlib.cm.get_cmap("jet")

        curvature = self._position_to_curvature.get(position)
        curvature_color = "gray"
        if curvature is not None:
            curvature_scaled = (curvature - self._min_cell_curvature)/(self._max_cell_curvature - self._min_cell_curvature)
            curvature_color = color_map(min(1, curvature_scaled))
        if dt == 0 and abs(dz) <= 3:
            self._draw_selection(position, curvature_color)
