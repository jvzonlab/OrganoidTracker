from typing import Optional, Set

from networkx import Graph

from autotrack.comparison.report import ComparisonReport, Category
from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.resolution import ImageResolution
from autotrack.linking.existing_connections import find_future_particles
from autotrack.linking_analysis import cell_appearance_finder


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


def _find_closest_in(all_particles: Set[Particle], search: Particle, max_distance_um: float,
                     resolution: ImageResolution) -> Optional[Particle]:
    closest_particle = None
    closest_distance = float("inf")
    for particle in all_particles:
        if particle.time_point_number() != search.time_point_number():
            continue
        distance = particle.distance_um(search, resolution)
        if distance < closest_distance and distance < max_distance_um:
            closest_distance = distance
            closest_particle = particle
    return closest_particle


class _Comparing:

    _resolution: ImageResolution
    _max_distance_um: float
    _ground_truth: Graph
    _scratch: Graph

    def __init__(self, resolution: ImageResolution, ground_truth: Graph, scratch: Graph, max_distance_um: float):
        """Creates the comparison object. You need to provide two data sets. Cells are not allowed to move further away
         from each other than max_distance_um."""
        self._resolution = resolution
        self._ground_truth = ground_truth
        self._scratch = scratch
        self._max_distance_um = max_distance_um

    def compare_lineages(self, report: ComparisonReport, particle_ground_truth: Particle, particle_scratch: Particle):
        while True:
            next_ground_truth = list(find_future_particles(self._ground_truth, particle_ground_truth))
            next_scratch = list(find_future_particles(self._scratch, particle_scratch))
            if len(next_ground_truth) == 0:
                if len(next_scratch) != 0:
                    report.add_data(LINEAGE_END_FALSE_NEGATIVES, particle_ground_truth)
                else:
                    report.add_data(LINEAGE_END_TRUE_POSITIVES, particle_ground_truth)
                return

            if len(next_ground_truth) > 1:
                if len(next_scratch) != 2:
                    report.add_data(DIVISIONS_FALSE_NEGATIVES, particle_ground_truth)
                else:  # So both have len 2
                    report.add_data(DIVISIONS_TRUE_POSITIVES, particle_ground_truth)
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
                report.add_data(DIVISIONS_FALSE_POSITIVES, particle_ground_truth, "moves to", next_ground_truth[0],
                                "but was detected as dividing into", next_scratch[0], "and", next_scratch[1])
                return
            elif len(next_scratch) == 0:
                report.add_data(LINEAGE_END_FALSE_POSITIVES, particle_ground_truth)
                return

            # Both have length 1, continue looking in this lineage
            particle_ground_truth = next_ground_truth[0]
            particle_scratch = next_scratch[0]

            # If the detection data skipped time points, do the same for the ground truth data
            while particle_scratch.time_point_number() > particle_ground_truth.time_point_number():
                next_ground_truth = find_future_particles(self._ground_truth, particle_ground_truth)
                if len(next_ground_truth) == 0:  # Detection data skipped past a lineage end
                    report.add_data(LINEAGE_END_FALSE_NEGATIVES, particle_ground_truth)
                    return
                elif len(next_ground_truth) > 1:  # Detection data skipped past a cell division
                    report.add_data(DIVISIONS_FALSE_NEGATIVES, particle_ground_truth)
                    return
                else:
                    particle_ground_truth = next_ground_truth.pop()

            # Check distances
            distance_um = particle_ground_truth.distance_um(particle_scratch, self._resolution)
            if distance_um > self._max_distance_um:
                report.add_data(MOVEMENT_DISAGREEMENT, particle_ground_truth, "too far away from the detected position "
                                "at", particle_ground_truth, f"- difference is {distance_um:0.1f} um")
                return
            report.add_data(MOVEMENT_TRUE_POSITIVES, particle_ground_truth)


def compare_links(ground_truth: Experiment, scratch: Experiment, max_distance_um: float = 5) -> ComparisonReport:
    if ground_truth.links.graph is None or scratch.links.graph is None:
        raise UserError("Linking data is missing", "One of the data sets has no linking data available.")

    report = ComparisonReport()
    report.title = "Links comparison"
    comparing = _Comparing(ground_truth.image_resolution(), ground_truth.links.graph, scratch.links.graph,
                           max_distance_um)
    lineage_starts_ground_truth = cell_appearance_finder.find_appeared_cells(ground_truth.links.graph)
    lineage_starts_scratch = set(cell_appearance_finder.find_appeared_cells(scratch.links.graph))
    for lineage_start_ground_truth in lineage_starts_ground_truth:
        lineage_start_scratch = _find_closest_in(lineage_starts_scratch, lineage_start_ground_truth, max_distance_um,
                                                 ground_truth.image_resolution())
        if lineage_start_scratch is None:
            report.add_data(LINEAGE_START_FALSE_NEGATIVES, lineage_start_ground_truth)
            continue
        # Make sure no other lineage can compare itself to this track
        lineage_starts_scratch.remove(lineage_start_scratch)
        report.add_data(LINEAGE_START_TRUE_POSITIVES, lineage_start_ground_truth)
        comparing.compare_lineages(report, lineage_start_ground_truth, lineage_start_scratch)
    return report
