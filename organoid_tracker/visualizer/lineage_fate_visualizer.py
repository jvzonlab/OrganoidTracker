from typing import Dict, Optional

import matplotlib.cm

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.linking_analysis import lineage_fate_finder
from organoid_tracker.linking_analysis.lineage_fate_finder import LineageFate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _lineage_fate_to_text(lineage_fate: Optional[LineageFate]):
    # Special cases
    if lineage_fate is None or lineage_fate.tracks == 0:
        return "?"
    if lineage_fate.deaths == 1 and lineage_fate.sheds == 0 and lineage_fate.divisions == 0:
        return "X"
    if lineage_fate.deaths == 0 and lineage_fate.sheds == 1 and lineage_fate.divisions == 0:
        return "S"

    # Normal cases
    values = []
    if lineage_fate.divisions > 0:
        values.append(str(lineage_fate.divisions))
    if lineage_fate.deaths > 0:
        values.append(f"{lineage_fate.deaths}X")
    if lineage_fate.sheds > 0:
        values.append(f"{lineage_fate.sheds}S")
    if len(values) > 0:
        return str.join(", ", values)

    # No interesting events
    if lineage_fate.ends > 0:
        return "~|"
    return "~"


class LineageFateVisualizer(ExitableImageVisualizer):
    """Shows how each cell will develop during the experiment. Colors represent the final position on the data axis.
    Label legend:
    ?   no reliable linking data available.
    X   cell died,   ~|   lineage ended for some other reason.
    4   cell divided four times. "4, 1X" means cell divided four times, one offspring cell died."
    ~   no events, just movement during the complete experiment."""

    _lineage_fates: Dict[Position, LineageFate] = dict()
    _position_to_axis: Dict[Position, float] = dict()
    _highest_axis_position: float = 1

    def _calculate_time_point_metadata(self):
        super()._calculate_time_point_metadata()

        self._update_deaths_and_divisions(self._time_point)
        self._update_axis_positions(self._time_point)

    def _update_deaths_and_divisions(self, time_point: TimePoint):
        # Check what lineages contain errors
        links = self._experiment.links
        positions = self._experiment.positions
        if not links.has_links():
            self._lineage_fates = dict()
            return

        positions = self._experiment.positions.of_time_point(time_point)
        links = self._experiment.links
        last_time_point_number = self._experiment.positions.last_time_point_number()
        result = dict()
        for position in positions:
            result[position] = lineage_fate_finder.get_lineage_fate(position, links, positions,
                                                                    last_time_point_number)
        self._lineage_fates = result

    def _update_axis_positions(self, time_point: TimePoint):
        max_position = 1
        self._position_to_axis = dict()
        for position in self._experiment.positions.of_time_point(time_point):
            axis_position = self._calculate_final_crypt_axis_position(position)
            if axis_position is not None:
                self._position_to_axis[position] = axis_position
                if axis_position > max_position:
                    max_position = axis_position
        self._highest_position = max_position

    def _calculate_final_crypt_axis_position(self, position: Position) -> Optional[float]:
        links = self._experiment.links
        axes = self._experiment.splines

        track = links.get_track(position)
        if track is None:
            return None

        # Calculate average (total/count) the crypt position of all final positions
        crypt_axis_position_total = 0
        crypt_axis_position_count = 0
        for descending_track in track.find_all_descending_tracks(include_self=True):
            if descending_track.get_next_tracks():
                continue  # This is not a final track (but for example a dividing track), ignore

            last_position = descending_track.find_last_position()
            crypt_axis_position = axes.to_position_on_original_axis(links, last_position)
            if crypt_axis_position is None:
                # No crypt axes defined for this time point
                continue

            crypt_axis_position_total += crypt_axis_position.pos
            crypt_axis_position_count += 1

        if crypt_axis_position_count == 0:
            return None  # No crypt axes recorded
        return crypt_axis_position_total / crypt_axis_position_count

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if dt != 0 or abs(dz) > 3:
            return True

        lineage_fate = self._lineage_fates.get(position)
        background_color = (0.2, 0.2, 0.2)
        try:
            axis_fraction = self._position_to_axis[position] / self._highest_position
            background_color = matplotlib.cm.jet(axis_fraction)
        except KeyError:
            pass  # Not interesting
        text_color = "black" if sum(background_color) / len(background_color) > 0.6 else "white"
        self._draw_annotation(position, _lineage_fate_to_text(lineage_fate), text_color=text_color,
                              background_color=background_color)
        return True
