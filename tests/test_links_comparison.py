import unittest
from time import sleep

from networkx import Graph

from autotrack.comparison import links_comparison
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.core.resolution import ImageResolution


def _experiment(*particles: Particle) -> Experiment:
    """Creates a testing experiment containing the given particles. Resolution is simply 1px = 1um"""
    experiment = Experiment()
    experiment.image_resolution(ImageResolution(1, 1, 1, 1))  # Set 1 px = 1 um for simplicity
    experiment.links.set_links(Graph())
    for particle in particles:
        experiment.add_particle(particle)
    return experiment


class TestLinksComparison(unittest.TestCase):

    def test_movement_reports(self):
        # Create a data set with one particle that moves twice to the right: a1 -> a2 -> a2
        a1 = Particle(0, 0, 0).with_time_point_number(1)
        a2 = Particle(1, 0, 0).with_time_point_number(2)
        a3 = Particle(2, 0, 0).with_time_point_number(3)
        ground_truth = _experiment(a1, a2, a3)
        ground_truth.links.graph.add_edge(a1, a2)
        ground_truth.links.graph.add_edge(a2, a3)

        # Create another data set that does the same, but 10px apart: b1 -> b2 -> b3
        b1 = Particle(1, 0, 0).with_time_point_number(1)
        b2 = Particle(2, 0, 0).with_time_point_number(2)
        b3 = Particle(12, 0, 0).with_time_point_number(3)  # Simulate rapid movement, which may indicate mistracking
        scratch = _experiment(b1, b2, b3)
        scratch.links.graph.add_edge(b1, b2)
        scratch.links.graph.add_edge(b2, b3)

        result = links_comparison.compare_links(ground_truth, scratch, max_distance_um=11)
        self.assertEquals(1, result.count(links_comparison.LINEAGE_START_TRUE_POSITIVES))
        self.assertEquals(0, result.count(links_comparison.LINEAGE_START_FALSE_NEGATIVES))
        self.assertEquals(2, result.count(links_comparison.MOVEMENT_TRUE_POSITIVES))
        self.assertEquals(0, result.count(links_comparison.MOVEMENT_DISAGREEMENT))

        result_lower_max_distance = links_comparison.compare_links(ground_truth, scratch, max_distance_um=5)
        self.assertEquals(1, result_lower_max_distance.count(links_comparison.LINEAGE_START_TRUE_POSITIVES))
        self.assertEquals(0, result_lower_max_distance.count(links_comparison.LINEAGE_START_FALSE_NEGATIVES))
        self.assertEquals(1, result_lower_max_distance.count(links_comparison.MOVEMENT_TRUE_POSITIVES))
        self.assertEquals(1, result_lower_max_distance.count(links_comparison.MOVEMENT_DISAGREEMENT))

    def test_missed_division(self):
        # Create a cell division; a1 -> [a2, a3]
        a1 = Particle(0, 0, 0).with_time_point_number(1)
        a2 = Particle(-1, 0, 0).with_time_point_number(2)
        a3 = Particle(1, 0, 0).with_time_point_number(2)
        ground_truth = _experiment(a1, a2, a3)
        ground_truth.links.graph.add_edge(a1, a2)
        ground_truth.links.graph.add_edge(a1, a3)

        # Create a missed cell division; b1 -> b2, but b3 has no links
        b1 = Particle(0, 0, 0).with_time_point_number(1)
        b2 = Particle(-1, 0, 0).with_time_point_number(2)
        b3 = Particle(1, 0, 0).with_time_point_number(2)
        scratch = _experiment(b1, b2, b3)
        scratch.links.graph.add_edge(b1, b2)

        result = links_comparison.compare_links(ground_truth, scratch)
        self.assertEquals(1, result.count(links_comparison.DIVISIONS_FALSE_NEGATIVES))
        self.assertEquals(0, result.count(links_comparison.DIVISIONS_TRUE_POSITIVES))
        self.assertEquals(0, result.count(links_comparison.DIVISIONS_FALSE_POSITIVES))

    def test_one_track_missed(self):
        # Ground truth has two cell tracks: a1 -> a2 -> a3, a101 -> a102, a103
        a1 = Particle(0, 0, 0).with_time_point_number(1)
        a2 = Particle(1, 0, 0).with_time_point_number(2)
        a3 = Particle(2, 0, 0).with_time_point_number(3)
        a101 = Particle(10, 0, 0).with_time_point_number(1)
        a102 = Particle(11, 0, 0).with_time_point_number(2)
        a103 = Particle(12, 0, 0).with_time_point_number(3)
        ground_truth = _experiment(a1, a2, a3, a101, a102, a103)
        ground_truth.links.graph.add_edges_from([(a1, a2), (a2, a3)])
        ground_truth.links.graph.add_edges_from([(a101, a102), (a102, a103)])

        # Scratch data has only one cell track: b1 -> b2 -> b3
        b1 = Particle(5, 0, 0).with_time_point_number(1)
        b2 = Particle(6, 0, 0).with_time_point_number(2)
        b3 = Particle(7, 0, 0).with_time_point_number(3)
        scratch = _experiment(b1, b2, b3)
        scratch.links.graph.add_edges_from([(b1, b2), (b2, b3)])

        result = links_comparison.compare_links(ground_truth, scratch, max_distance_um=8)
        self.assertEquals(1, result.count(links_comparison.LINEAGE_START_TRUE_POSITIVES))
        self.assertEquals(1, result.count(links_comparison.LINEAGE_START_FALSE_NEGATIVES))
