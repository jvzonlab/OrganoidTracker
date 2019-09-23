"""Used to color cells by their lineage."""
from typing import Dict, Any

from matplotlib.backend_bases import MouseEvent
import matplotlib.colors

from ai_track.core.position import Position
from ai_track.gui.window import Window
from ai_track.visualizer import activate
from ai_track.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "View//Tracks-Cell colored by lineage...": lambda: _view_cells_colored_by_lineage(window)
    }


def _view_cells_colored_by_lineage(window: Window):
    activate(_CellsColoredByLineageVisualizer(window))


class _CellsColoredByLineageVisualizer(ExitableImageVisualizer):
    """Colors each cell by its lineage: cells with the same color share a common ancestor."""

    def _on_mouse_click(self, event: MouseEvent):
        if event.dblclick and event.xdata is not None and event.ydata is not None:
            position = self._get_position_at(event.xdata, event.ydata)
            if position is not None:
                from ai_track.linking_analysis import lineage_id_creator
                links = self._experiment.links
                lineage_id = lineage_id_creator.get_lineage_id(links, position)
                color = lineage_id_creator.get_color_for_lineage_id(lineage_id)
                color_str = matplotlib.colors.to_hex(color, keep_alpha=False)

                self.update_status(f"Lineage id of {position} is {lineage_id} (color is {color_str}).")

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        from ai_track.linking_analysis import lineage_id_creator

        if abs(dz) > self.MAX_Z_DISTANCE or dt != 0:
            return super()._on_position_draw(position, color, dz, dt)

        links = self._experiment.links
        lineage_id = lineage_id_creator.get_lineage_id(links, position)
        if lineage_id == -1:
            return super()._on_position_draw(position, color, dz, dt)

        lineage_color = lineage_id_creator.get_color_for_lineage_id(lineage_id)
        self._draw_selection(position, lineage_color)
        return super()._on_position_draw(position, color, dz, dt)
