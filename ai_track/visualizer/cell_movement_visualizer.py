from matplotlib import cm

from ai_track.core import COLOR_CELL_CURRENT
from ai_track.core.position import Position
from ai_track.imaging import angles
from ai_track.linking_analysis import particle_movement_finder
from ai_track.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class CellMovementVisualizer(ExitableImageVisualizer):
    """Shows how each cell move during the experiment using arrows."""

    def _draw_links(self):
        pass  # Don't draw links

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if abs(dz) > self.MAX_Z_DISTANCE + 1 or dt != 0:
            return False

        last_time_point = self._experiment.get_time_point(self._experiment.positions.last_time_point_number())
        future_positions = particle_movement_finder.find_future_positions_at(self._experiment.links, position,
                                                                             last_time_point)
        if len(future_positions) > 0:
            self._ax.scatter(position.x, position.y, marker='o', facecolor=COLOR_CELL_CURRENT,
                             edgecolors="black", s=17 ** 2, linewidths=2)
        for future_position in future_positions:
            if abs(future_position.x - position.x) < 2 and abs(future_position.y - position.y) < 2:
                # Distance is smaller than 2 px, don't draw
                continue
            self._ax.arrow(position.x, position.y,
                           future_position.x - position.x, future_position.y - position.y,
                           length_includes_head=True, width=5.5, head_width=9, facecolor=COLOR_CELL_CURRENT,
                           edgecolor="black", linewidth=2)
        return False  # Prevents drawing the usual dots
