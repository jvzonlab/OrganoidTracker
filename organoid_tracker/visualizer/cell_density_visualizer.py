from typing import Optional, Dict

import matplotlib

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.position_analysis import cell_density_calculator
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class CellDensityVisualizer(ExitableImageVisualizer):
    """Shows the density around a cell: how many cells are there in an area?"""

    _min_cell_density: Optional[float] = None
    _max_cell_density: Optional[float] = None
    _position_to_density: Dict[Position, float]

    def __init__(self, window: Window):
        super().__init__(window)

        self._position_to_density = dict()

    def _calculate_time_point_metadata(self):
        super()._calculate_time_point_metadata()
        self._calculate_densities()

    def _calculate_densities(self):
        print("Hi, calculating!")
        positions = self._experiment.positions.of_time_point(self._time_point)
        resolution = self._experiment.images.resolution()
        min_density = None
        max_density = None
        densities = dict()

        for position in positions:
            cell_density = cell_density_calculator.get_density_mm1(positions, around=position, resolution=resolution)
            if min_density is None or cell_density < min_density:
                min_density = cell_density
            if max_density is None or cell_density > max_density:
                max_density = cell_density
            densities[position] = cell_density

        self._min_cell_density = min_density
        self._max_cell_density = max_density
        self._position_to_density = densities

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        color_map = matplotlib.cm.get_cmap("jet")

        density = self._position_to_density.get(position)
        density_color = "gray"
        if density is not None:
            density_color = color_map((density - self._min_cell_density)/(self._max_cell_density - self._min_cell_density))
        if dt == 0 and abs(dz) <= 3:
            self._draw_selection(position, density_color)
        return True
