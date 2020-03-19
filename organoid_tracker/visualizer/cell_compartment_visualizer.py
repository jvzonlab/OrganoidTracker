from typing import Dict, Optional

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.linking_analysis import cell_fate_finder, cell_compartment_finder
from organoid_tracker.linking_analysis.cell_compartment_finder import CellCompartment
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType, CellFate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _cell_compartment_to_color(compartment: CellCompartment) -> MPLColor:
    if compartment == CellCompartment.DIVIDING:
        return "#55efc4"
    if compartment == CellCompartment.NON_DIVIDING:
        return "#0984e3"
    return "#2d3436"


class CellCompartmentVisualizer(ExitableImageVisualizer):
    """Divides the tissue into a dividing and a non-dividing compartment. A cell belongs to the dividing compartment
    (green) if it divides, or one of its neighbors divides. A cell belongs to the non-dividing compartment (blue) if we
    know that cell does not divide, and no divisions are found in the neighbors. Other cells are drawn in black."""

    _cell_compartments: Dict[Position, CellCompartment] = dict()

    def _calculate_time_point_metadata(self):
        super()._calculate_time_point_metadata()

        # Check what lineages contain errors
        links = self._experiment.links
        if not links.has_links():
            self._cell_compartments = dict()
            return

        positions = self._experiment.positions.of_time_point(self._time_point)
        result = dict()
        for position in positions:
            result[position] = cell_compartment_finder.find_compartment(self._experiment, position)
        self._cell_compartments = result

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if dt == 0 and abs(dz) <= 3:
            cell_compartment = self._cell_compartments.get(position, CellCompartment.UNKNOWN)
            color = _cell_compartment_to_color(cell_compartment)
            self._draw_selection(position, color)
        return True
