from typing import Dict, Optional

from ai_track.core import TimePoint
from ai_track.core.position import Position
from ai_track.core.typing import MPLColor
from ai_track.linking_analysis import cell_fate_finder, cell_compartment_finder
from ai_track.linking_analysis.cell_compartment_finder import CellCompartment
from ai_track.linking_analysis.cell_fate_finder import CellFateType, CellFate
from ai_track.visualizer.exitable_image_visualizer import ExitableImageVisualizer


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

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)

        # Check what lineages contain errors
        links = self._experiment.links
        if not links.has_links():
            self._cell_compartments = dict()
            return

        positions = self._experiment.positions.of_time_point(time_point)
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
