from typing import Dict, Any, Optional

from organoid_tracker.core.position import Position
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _division_score_to_color(division_score: float):
    if division_score is None or division_score > 0:
        return "black"
    else:
        return "lime"


class DivisionScoreVisualizer(ExitableImageVisualizer):
    """Shows the division scores for each position at the current time point. In the View menu, you can toggle between
    showing the raw division score and the likelihood percentage.

    In case you haven't predicted the division scores yet, a "?" is shown for all positions. (To predict division
    scores, run the cell tracker from the Tools menu in the main screen.)"""

    _division_scores: Dict[Position, float] = dict()
    _show_percentage: bool = True

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View//Divisions-Toggle showing percentage": self._toggle_show_percentage,
        }

    def _toggle_show_percentage(self):
        self._show_percentage = not self._show_percentage
        self.draw_view()

    def _calculate_time_point_metadata(self):
        super()._calculate_time_point_metadata()

        positions = self._experiment.positions.of_time_point(self._time_point)
        result = dict()
        for position in positions:
            result[position] = self._experiment.positions.get_position_data(position, 'division_penalty')
        self._division_scores = result

    def _division_score_to_text(self, division_score: float) -> str:
        if division_score is None:
            return '?'
        if self._show_percentage:
            likelihood = -division_score
            percentage = (10 ** likelihood) / (1 + 10 ** likelihood)
            if percentage < 0.51:
                return '<1%'  # Would get rounded to 0% otherwise
            return str(round(percentage * 100)) + '%'
        else:
            return str(round(division_score, 1))

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        division_score = self._division_scores.get(position)

        if dt == 0 and abs(dz) <= 3:
            text = self._division_score_to_text(division_score)
            color = _division_score_to_color(division_score)
            self._ax.annotate(text, (position.x, position.y), fontsize=10 - abs(dz / 2),
                              fontweight="bold", color=color, backgroundcolor=(1,1,1,0.4))
        return True

