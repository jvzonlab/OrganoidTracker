"""Compares two sets of links on the level of individual links. Missed cell divisions/deaths are not reported. Instead,
for every link in the baseline data, it is checked if it is present in the automatic data, and vice versa."""
from typing import Iterable

import math

from ai_track.comparison.report import Category, ComparisonReport, Statistics
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.linking import nearby_position_finder

LINKS_FALSE_NEGATIVES = Category("Missed links")
LINKS_TRUE_POSITIVES = Category("Correctly detected links")
LINKS_FALSE_POSITIVES = Category("Made up links")


class LinksReport(ComparisonReport):

    def __init__(self):
        super().__init__()
        self.title = "Links comparison"

    def calculate_correctness_statistics(self) -> Statistics:
        return self.calculate_statistics(LINKS_TRUE_POSITIVES, LINKS_FALSE_POSITIVES, LINKS_FALSE_NEGATIVES)


def _find_close_positions(position: Position, experiment: Experiment, max_distance: float) -> Iterable[Position]:
    all_positions = experiment.positions.of_time_point(position.time_point())
    resolution = experiment.images.resolution()
    return nearby_position_finder.find_closest_n_positions(all_positions, around=position, resolution=resolution,
                                                           max_amount=3, max_distance_um=max_distance, ignore_self=False)


def _has_link(links: Links, positions1: Iterable[Position], positions2: Iterable[Position]) -> bool:
    for position1 in positions1:
        for position2 in positions2:
            if links.contains_link(position1, position2):
                return True
    return False


def compare_links(ground_truth: Experiment, scratch: Experiment, max_distance_um: float = 5) -> LinksReport:
    result = LinksReport()

    # Check if all baseline links exist
    for position1, position2 in ground_truth.links.find_all_links():
        scratch_positions1 = _find_close_positions(position1, scratch, max_distance_um)
        scratch_positions2 = _find_close_positions(position2, scratch, max_distance_um)
        if _has_link(scratch.links, scratch_positions1, scratch_positions2):
            # True positive
            result.add_data(LINKS_TRUE_POSITIVES, position1, "is linked to", position2)
        else:
            # False negative
            result.add_data(LINKS_FALSE_NEGATIVES, position1, "has mistakenly no link to", position2)

    # Check if all scratch links are real
    for position1, position2 in scratch.links.find_all_links():
        ground_thruth_positions1 = _find_close_positions(position1, ground_truth, max_distance_um)
        ground_thruth_positions2 = _find_close_positions(position2, ground_truth, max_distance_um)
        if _has_link(ground_truth.links, ground_thruth_positions1, ground_thruth_positions2):
            # True positive, already detected in earlier loop
            pass
        else:
            # False positive
            result.add_data(LINKS_FALSE_POSITIVES, position1, "has mistakenly a link to", position2)

    return result
