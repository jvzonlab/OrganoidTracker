from typing import Dict, Optional, Any

from matplotlib.collections import LineCollection

from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _link_score_to_color(link_score: Optional[float]) -> str:
    if link_score is None or link_score > 0:
        return "black"
    else:
        return "lime"


def _to_linewidth(link_score: Optional[float]) -> float:
    if link_score is None:
        return 1
    linewidth = 2 - link_score
    if linewidth < 1:
        return 1
    if linewidth > 6:
        return 6
    return linewidth


class LinkScoreVisualizer(ExitableImageVisualizer):
    """Shows the link score for each link at the current time point. In the View menu, you can toggle between
    showing the raw link score (lower = more likely) and the likelihood percentage.

    In case you haven't predicted the link scores yet, a "?" is shown for all links. (To predict link
    scores, run the cell tracker from the Tools menu in the main screen.)"""

    _show_percentage: bool = True

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View//Links-Toggle showing percentage": self._toggle_show_percentage,
        }

    def _toggle_show_percentage(self):
        self._show_percentage = not self._show_percentage
        self.draw_view()

    def _must_draw_positions_of_previous_time_point(self) -> bool:
        # We ignore links to previous time point in this visualizer, this makes the display less cluttered
        return False

    def _draw_links(self):
        if not self._display_settings.show_links_and_connections:
            return
        max_intensity_projection = self._display_settings.max_intensity_projection

        lines = []
        colors = []
        linewidths = []
        for position1, position2 in self._experiment.links.of_time_point(self._time_point):
            min_display_z = min(position1.z, position2.z) - self.MAX_Z_DISTANCE
            max_display_z = max(position1.z, position2.z) + self.MAX_Z_DISTANCE
            if (self._z < min_display_z or self._z > max_display_z) and not max_intensity_projection:
                continue
            if position2.time_point_number() < position1.time_point_number():
                # Ignore links to previous time point in this visualizer, this makes the display less cluttered
                continue

            line = (position1.x, position1.y), (position2.x, position2.y)
            lines.append(line)

            link_score = self._experiment.links.get_link_data(position1, position2, "link_penalty")
            color = _link_score_to_color(link_score)
            colors.append(color)

            linewidths.append(_to_linewidth(link_score))
            dz = int(abs((position1.z + position2.z)/2 - self._z))
            if max_intensity_projection:
                dz = 0
            self._ax.annotate(self._link_score_to_text(link_score),
                              ((position1.x + position2.x)/2, (position1.y + position2.y)/2),
                              fontsize=10 - abs(dz / 2),
                              fontweight="bold",
                              color="black",
                              backgroundcolor=(1, 1, 1, 0.4))

        self._ax.add_collection(LineCollection(lines, colors=colors, linewidths=linewidths))

    def _link_score_to_text(self, link_score: Optional[float]) -> str:
        if link_score is None:
            return '?'
        if self._show_percentage:
            likelihood = -link_score
            percentage = (10 ** likelihood) / (1 + 10 ** likelihood)
            if percentage < 0.51:
                return '<1%'  # Would get rounded to 0% otherwise
            return str(round(percentage * 100)) + '%'
        else:
            return str(round(link_score, 1))
