from typing import Dict
from organoid_tracker.core.position import Position
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _link_score_to_text(link_score: float):
    if link_score is None:
        return '?'
    else:
        return str(round(link_score, 1))


def _division_score_to_color(link_score: float):
    if link_score is None or link_score > 0:
        return "black"
    else:
        return "red"


class LinkScoreVisualizer(ExitableImageVisualizer):

    _link_scores: Dict[Position, float] = dict()

    def _calculate_time_point_metadata(self):
        super()._calculate_time_point_metadata()

        positions = self._experiment.positions.of_time_point(self._time_point)
        result = dict()
        for position in positions:
            next_positions = list(self._experiment.links.find_futures(position))
            if next_positions is not None and len(next_positions)>0:
                print(next_positions)
                next_positions = next_positions[0]
                print(self._experiment.link_data.get_link_data(position, next_positions, 'link_penalty'))
                result[position] = self._experiment.link_data.get_link_data(position, next_positions, 'link_penalty')
        self._link_scores = result

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        link_score = self._link_scores.get(position)

        if dt == 0 and abs(dz) <= 3:
            color = _division_score_to_color(link_score)
            self._ax.annotate(_link_score_to_text(link_score), (position.x, position.y), fontsize=8 - abs(dz / 2),
                              fontweight="bold", color=color, backgroundcolor=(1,1,1,0.2))
        return True

