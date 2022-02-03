from typing import Dict
from organoid_tracker.core.position import Position
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _division_score_to_text(division_score: float):
    if division_score is None:
        return '?'
    else:
        return str(round(division_score, 1))


def _division_score_to_color(division_score: float):
    if division_score is None or division_score > 0:
        return "black"
    else:
        return "lime"


class DivisionScoreVisualizer(ExitableImageVisualizer):

    _division_scores: Dict[Position, float] = dict()

    def _calculate_time_point_metadata(self):
        super()._calculate_time_point_metadata()

        positions = self._experiment.positions.of_time_point(self._time_point)
        result = dict()
        for position in positions:
            result[position] = self._experiment.position_data.get_position_data(position, 'division_penalty')
        self._division_scores = result

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        division_score = self._division_scores.get(position)

        if dt == 0 and abs(dz) <= 3:
            color = _division_score_to_color(division_score)
            self._ax.annotate(_division_score_to_text(division_score), (position.x, position.y), fontsize=10 - abs(dz / 2),
                              fontweight="bold", color=color, backgroundcolor=(1,1,1,0.4))
        return True

