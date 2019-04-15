"""Compares two sets of links on the level of individual links. Missed cell divisions/deaths are not reported. Instead,
for every link in the baseline data, it is checked if it is present in the automatic data, and vice versa."""
from typing import Iterable

import math

from autotrack.comparison.report import Category, ComparisonReport
from autotrack.core.experiment import Experiment
from autotrack.core.links import Links
from autotrack.core.position import Position
from autotrack.linking import nearby_position_finder

LINKS_FALSE_NEGATIVES = Category("Missed links")
LINKS_TRUE_POSITIVES = Category("Correctly detected links")
LINKS_FALSE_POSITIVES = Category("Made up links")


def _find_close_positions(position: Position, experiment: Experiment, max_distance: int) -> Iterable[Position]:
    all_positions = experiment.positions.of_time_point(position.time_point())
    return nearby_position_finder.find_closest_n_positions(all_positions, position,
                                                           max_amount=3, max_distance=max_distance, ignore_self=False)


def _has_link(links: Links, positions1: Iterable[Position], positions2: Iterable[Position]) -> bool:
    for position1 in positions1:
        for position2 in positions2:
            if links.contains_link(position1, position2):
                return True
    return False


def compare_links(ground_truth: Experiment, scratch: Experiment, max_distance_um: float = 5) -> ComparisonReport:
    result = ComparisonReport()

    # Check if all baseline links exist
    max_distance = math.ceil(max_distance_um / scratch.images.resolution().pixel_size_x_um)
    for position1, position2 in ground_truth.links.find_all_links():
        scratch_positions1 = _find_close_positions(position1, scratch, max_distance)
        scratch_positions2 = _find_close_positions(position2, scratch, max_distance)
        if _has_link(scratch.links, scratch_positions1, scratch_positions2):
            # True positive
            result.add_data(LINKS_TRUE_POSITIVES, position1, "is linked to", position2)
        else:
            # False negative
            result.add_data(LINKS_FALSE_NEGATIVES, position1, "has mistakenly no link to", position2)

    # Check if all scratch links are real
    # ... false positives are not yet reported!

    return result
