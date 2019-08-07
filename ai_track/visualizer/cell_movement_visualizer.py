from matplotlib import cm

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
        colormap = cm.get_cmap('hsv')
        for future_position in future_positions:
            if abs(future_position.x - position.x) < 2 and abs(future_position.y - position.y) < 2:
                # Distance is smaller than 2 px, don't draw
                continue
            direction = angles.direction_2d(position, future_position)
            color = colormap(direction / 360)
            self._ax.arrow(position.x, position.y, future_position.x - position.x, future_position.y - position.y,
                           length_includes_head=True, width=3, color=color)
        return False  # Prevents drawing the usual dots
