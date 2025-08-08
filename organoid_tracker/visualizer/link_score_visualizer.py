from typing import Dict

from matplotlib.collections import LineCollection

from organoid_tracker import core
from organoid_tracker.core.position import Position
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _link_score_to_text(link_score: float):
    if link_score is None:
        return '?'
    else:
        return str(round(link_score, 1))


def _link_score_to_color(link_score: float):
    if link_score is None or link_score > 0:
        return "black"
    else:
        return "lime"


def _to_linewidth(link_score: float):
    linewidth = 2 - link_score
    if linewidth < 1:
        return 1
    if linewidth > 6:
        return 6
    return linewidth


class LinkScoreVisualizer(ExitableImageVisualizer):
    """Shows the penalty score of each link. The lower the score, the more likely that the network thinks the link is
    correct."""

    def _must_draw_positions_of_previous_time_point(self) -> bool:
        # We ignore links to previous time point in this visualizer, this makes the display less cluttered
        return False

    def _draw_links(self):
        if not self._display_settings.show_links_and_connections:
            return

        lines = []
        colors = []
        linewidths = []
        for position1, position2 in self._experiment.links.of_time_point(self._time_point):
            min_display_z = min(position1.z, position2.z) - self.MAX_Z_DISTANCE
            max_display_z = max(position1.z, position2.z) + self.MAX_Z_DISTANCE
            if self._z < min_display_z or self._z > max_display_z:
                continue
            if position2.time_point_number() < position1.time_point_number():
                # Ignore links to previous time point in this visualizer, this makes the display less cluttered
                continue

            line = (position1.x, position1.y), (position2.x, position2.y)
            lines.append(line)

            link_score = self._experiment.links.get_link_data(position1, position2, "link_penalty")
            if link_score is None:
                link_score = 0
            color = _link_score_to_color(link_score)
            colors.append(color)

            linewidths.append(_to_linewidth(link_score))
            dz = int(abs((position1.z + position2.z)/2 - self._z))
            self._ax.annotate(_link_score_to_text(link_score),
                              ((position1.x + position2.x)/2, (position1.y + position2.y)/2),
                              fontsize=10 - abs(dz / 2),
                              fontweight="bold",
                              color="black",
                              backgroundcolor=(1, 1, 1, 0.4))

        self._ax.add_collection(LineCollection(lines, colors=colors, linewidths=linewidths))

