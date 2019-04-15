import unittest

from autotrack.comparison import lineage_comparison
from autotrack.core.experiment import Experiment
from autotrack.core.position import Position
from autotrack.core.resolution import ImageResolution


def _experiment(*positions: Position) -> Experiment:
    """Creates a testing experiment containing the given positions. Resolution is simply 1px = 1um"""
    experiment = Experiment()
    experiment.images.set_resolution(ImageResolution(1, 1, 1, 1))  # Set 1 px = 1 um for simplicity
    for position in positions:
        experiment.positions.add(position)
    return experiment


class TestLinksComparison(unittest.TestCase):

    def test_movement_reports(self):
        # Create a data set with one position that moves twice to the right: a1 -> a2 -> a2
        a1 = Position(0, 0, 0, time_point_number=1)
        a2 = Position(1, 0, 0, time_point_number=2)
        a3 = Position(2, 0, 0, time_point_number=3)
        ground_truth = _experiment(a1, a2, a3)
        ground_truth.links.add_link(a1, a2)
        ground_truth.links.add_link(a2, a3)
        ground_truth.links.debug_sanity_check()

        # Create another data set that does the same, but 10px apart: b1 -> b2 -> b3
        b1 = Position(1, 0, 0, time_point_number=1)
        b2 = Position(2, 0, 0, time_point_number=2)
        b3 = Position(12, 0, 0, time_point_number=3)  # Simulate rapid movement, which may indicate mistracking
        scratch = _experiment(b1, b2, b3)
        scratch.links.add_link(b1, b2)
        scratch.links.add_link(b2, b3)

        result = lineage_comparison.compare_links(ground_truth, scratch, max_distance_um=11)
        self.assertEquals(1, result.count(lineage_comparison.LINEAGE_START_TRUE_POSITIVES))
        self.assertEquals(0, result.count(lineage_comparison.LINEAGE_START_FALSE_NEGATIVES))
        self.assertEquals(2, result.count(lineage_comparison.MOVEMENT_TRUE_POSITIVES))
        self.assertEquals(0, result.count(lineage_comparison.MOVEMENT_DISAGREEMENT))

        result_lower_max_distance = lineage_comparison.compare_links(ground_truth, scratch, max_distance_um=5)
        self.assertEquals(1, result_lower_max_distance.count(lineage_comparison.LINEAGE_START_TRUE_POSITIVES))
        self.assertEquals(0, result_lower_max_distance.count(lineage_comparison.LINEAGE_START_FALSE_NEGATIVES))
        self.assertEquals(1, result_lower_max_distance.count(lineage_comparison.MOVEMENT_TRUE_POSITIVES))
        self.assertEquals(1, result_lower_max_distance.count(lineage_comparison.MOVEMENT_DISAGREEMENT))

    def test_missed_division(self):
        # Create a cell division; a1 -> [a2, a3]
        a1 = Position(0, 0, 0, time_point_number=1)
        a2 = Position(-1, 0, 0, time_point_number=2)
        a3 = Position(1, 0, 0, time_point_number=2)
        ground_truth = _experiment(a1, a2, a3)
        ground_truth.links.add_link(a1, a2)
        ground_truth.links.add_link(a1, a3)
        ground_truth.links.debug_sanity_check()

        # Create a missed cell division; b1 -> b2, but b3 has no links
        b1 = Position(0, 0, 0, time_point_number=1)
        b2 = Position(-1, 0, 0, time_point_number=2)
        b3 = Position(1, 0, 0, time_point_number=2)
        scratch = _experiment(b1, b2, b3)
        scratch.links.add_link(b1, b2)

        result = lineage_comparison.compare_links(ground_truth, scratch)
        self.assertEquals(1, result.count(lineage_comparison.DIVISIONS_FALSE_NEGATIVES))
        self.assertEquals(0, result.count(lineage_comparison.DIVISIONS_TRUE_POSITIVES))
        self.assertEquals(0, result.count(lineage_comparison.DIVISIONS_FALSE_POSITIVES))

    def test_one_track_missed(self):
        # Ground truth has two cell tracks: a1 -> a2 -> a3, a101 -> a102, a103
        a1 = Position(0, 0, 0, time_point_number=1)
        a2 = Position(1, 0, 0, time_point_number=2)
        a3 = Position(2, 0, 0, time_point_number=3)
        a101 = Position(10, 0, 0, time_point_number=1)
        a102 = Position(11, 0, 0, time_point_number=2)
        a103 = Position(12, 0, 0, time_point_number=3)
        ground_truth = _experiment(a1, a2, a3, a101, a102, a103)
        ground_truth.links.add_link(a1, a2)
        ground_truth.links.add_link(a2, a3)
        ground_truth.links.add_link(a101, a102)
        ground_truth.links.add_link(a102, a103)

        # Scratch data has only one cell track: b1 -> b2 -> b3
        b1 = Position(5, 0, 0, time_point_number=1)
        b2 = Position(6, 0, 0, time_point_number=2)
        b3 = Position(7, 0, 0, time_point_number=3)
        scratch = _experiment(b1, b2, b3)
        scratch.links.add_link(b1, b2)
        scratch.links.add_link(b2, b3)

        result = lineage_comparison.compare_links(ground_truth, scratch, max_distance_um=8)
        self.assertEquals(1, result.count(lineage_comparison.LINEAGE_START_TRUE_POSITIVES))
        self.assertEquals(1, result.count(lineage_comparison.LINEAGE_START_FALSE_NEGATIVES))
