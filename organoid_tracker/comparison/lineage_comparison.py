"""Compares two sets of links on the lineage level. Things like missed cell divisions and cell deaths will be
reported."""

from typing import Optional, Set

from organoid_tracker.comparison.report import ComparisonReport, Category
from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution

LINEAGE_END_FALSE_NEGATIVES = Category("Missed lineage ends")
LINEAGE_END_TRUE_POSITIVES = Category("Correctly detected lineage ends")
LINEAGE_END_FALSE_POSITIVES = Category("Made up lineage ends")
DIVISIONS_FALSE_NEGATIVES = Category("Missed cell division")
DIVISIONS_TRUE_POSITIVES = Category("Correctly detected cell divisions")
DIVISIONS_FALSE_POSITIVES = Category("Made up cell divisions")
MOVEMENT_DISAGREEMENT = Category("Distance between cells became too large")
MOVEMENT_TRUE_POSITIVES = Category("Correctly detected moving cells")
LINEAGE_START_FALSE_NEGATIVES = Category("Missed lineage starts")
LINEAGE_START_TRUE_POSITIVES = Category("Correctly detected lineage starts")


def _find_closest_in(all_positions: Set[Position], search: Position, max_distance_um: float,
                     resolution: ImageResolution) -> Optional[Position]:
    closest_position = None
    closest_distance = float("inf")
    for position in all_positions:
        if position.time_point_number() != search.time_point_number():
            continue
        distance = position.distance_um(search, resolution)
        if distance < closest_distance and distance < max_distance_um:
            closest_distance = distance
            closest_position = position
    return closest_position


class _Comparing:

    _resolution: ImageResolution
    _max_distance_um: float
    _ground_truth: Links
    _scratch: Links

    def __init__(self, resolution: ImageResolution, ground_truth: Links, scratch: Links, max_distance_um: float):
        """Creates the comparison object. You need to provide two data sets. Cells are not allowed to move further away
         from each other than max_distance_um."""
        self._resolution = resolution
        self._ground_truth = ground_truth
        self._scratch = scratch
        self._max_distance_um = max_distance_um

    def compare_lineages(self, report: ComparisonReport, position_ground_truth: Position, position_scratch: Position):
        while True:
            next_ground_truth = list(self._ground_truth.find_futures(position_ground_truth))
            next_scratch = list(self._scratch.find_futures(position_scratch))
            if len(next_ground_truth) == 0:
                if len(next_scratch) != 0:
                    report.add_data(LINEAGE_END_FALSE_NEGATIVES, position_ground_truth)
                else:
                    report.add_data(LINEAGE_END_TRUE_POSITIVES, position_ground_truth)
                return

            if len(next_ground_truth) > 1:
                if len(next_scratch) != 2:
                    report.add_data(DIVISIONS_FALSE_NEGATIVES, position_ground_truth)
                else:  # So both have len 2
                    report.add_data(DIVISIONS_TRUE_POSITIVES, position_ground_truth)
                    distance_one_one = next_ground_truth[0].distance_um(next_scratch[0], self._resolution)
                    distance_one_two = next_ground_truth[0].distance_um(next_scratch[1], self._resolution)
                    if distance_one_one < distance_one_two:
                        self.compare_lineages(report, next_ground_truth[0], next_scratch[0])
                        self.compare_lineages(report, next_ground_truth[1], next_scratch[1])
                    else:
                        self.compare_lineages(report, next_ground_truth[0], next_scratch[1])
                        self.compare_lineages(report, next_ground_truth[1], next_scratch[0])
                return

            # len(next_ground_truth) == 1
            if len(next_scratch) > 1:
                report.add_data(DIVISIONS_FALSE_POSITIVES, position_ground_truth, "moves to", next_ground_truth[0],
                                "but was detected as dividing into", next_scratch[0], "and", next_scratch[1])
                return
            elif len(next_scratch) == 0:
                report.add_data(LINEAGE_END_FALSE_POSITIVES, position_ground_truth)
                return

            # Both have length 1, continue looking in this lineage
            position_ground_truth = next_ground_truth[0]
            position_scratch = next_scratch[0]

            # If the detection data skipped time points, do the same for the ground truth data
            while position_scratch.time_point_number() > position_ground_truth.time_point_number():
                next_ground_truth = self._ground_truth.find_futures(position_ground_truth)
                if len(next_ground_truth) == 0:  # Detection data skipped past a lineage end
                    report.add_data(LINEAGE_END_FALSE_NEGATIVES, position_ground_truth)
                    return
                elif len(next_ground_truth) > 1:  # Detection data skipped past a cell division
                    report.add_data(DIVISIONS_FALSE_NEGATIVES, position_ground_truth)
                    return
                else:
                    position_ground_truth = next_ground_truth.pop()

            # Check distances
            distance_um = position_ground_truth.distance_um(position_scratch, self._resolution)
            if distance_um > self._max_distance_um:
                report.add_data(MOVEMENT_DISAGREEMENT, position_ground_truth, "too far away from the detected position "
                                "at", position_scratch, f"- difference is {distance_um:0.1f} um")
                return
            report.add_data(MOVEMENT_TRUE_POSITIVES, position_ground_truth)


def compare_links(ground_truth: Experiment, scratch: Experiment, max_distance_um: float = 5) -> ComparisonReport:
    """Compares two sets of links on the lineage level. Things like missed cell divisions and cell deaths will be
    reported."""
    if not ground_truth.links.has_links() or not scratch.links.has_links():
        raise UserError("Linking data is missing", "One of the data sets has no linking data available.")

    report = ComparisonReport()
    report.title = "Links comparison"
    comparing = _Comparing(ground_truth.images.resolution(), ground_truth.links, scratch.links, max_distance_um)
    lineage_starts_ground_truth = ground_truth.links.find_appeared_positions()
    lineage_starts_scratch = set(scratch.links.find_appeared_positions())
    for lineage_start_ground_truth in lineage_starts_ground_truth:
        lineage_start_scratch = _find_closest_in(lineage_starts_scratch, lineage_start_ground_truth, max_distance_um,
                                                 ground_truth.images.resolution())
        if lineage_start_scratch is None:
            report.add_data(LINEAGE_START_FALSE_NEGATIVES, lineage_start_ground_truth)
            continue
        # Make sure no other lineage can compare itself to this track
        lineage_starts_scratch.remove(lineage_start_scratch)
        report.add_data(LINEAGE_START_TRUE_POSITIVES, lineage_start_ground_truth)
        comparing.compare_lineages(report, lineage_start_ground_truth, lineage_start_scratch)
    return report
