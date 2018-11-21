from autotrack.comparison.report import ComparisonReport, Category, Statistics
from autotrack.core.experiment import Experiment
from autotrack.core.find_nearest_neighbors import find_nearest_particles


_DETECTIONS_FALSE_NEGATIVES = Category("Missed detections")
_DETECTIONS_TRUE_POSITIVES = Category("Found detections")
_DETECTIONS_FALSE_POSITIVES = Category("Made up detections")


class DetectionReport(ComparisonReport):

    def __init__(self):
        super().__init__()
        self.title = "Detection comparison"

    def calculate_detection_statistics(self) -> Statistics:
        return self.calculate_statistics(_DETECTIONS_TRUE_POSITIVES, _DETECTIONS_FALSE_POSITIVES,
                                         _DETECTIONS_FALSE_NEGATIVES)


def compare_positions(ground_truth: Experiment, scratch: Experiment, max_distance_um: float = 5) -> DetectionReport:
    """Checks how much the positions in the ground truth match with the given data."""
    resolution = ground_truth.image_resolution()

    report = DetectionReport()

    for time_point in ground_truth.time_points():
        baseline_particles = set(ground_truth.particles.of_time_point(time_point))
        scratch_particles = set(scratch.particles.of_time_point(time_point))
        for baseline_particle in baseline_particles:
            nearest_in_scratch = find_nearest_particles(scratch_particles, around=baseline_particle, tolerance=1)
            if len(nearest_in_scratch) == 0:
                report.add_data(_DETECTIONS_FALSE_NEGATIVES, baseline_particle, "No candidates in the scratch data left.")
                continue
            distance_um = baseline_particle.distance_um(nearest_in_scratch[0], resolution)
            if distance_um > max_distance_um:
                report.add_data(_DETECTIONS_FALSE_NEGATIVES, baseline_particle, f"Nearest cell was {distance_um:0.1f} um away")
                continue
            scratch_particles.remove(nearest_in_scratch[0])
            report.add_data(_DETECTIONS_TRUE_POSITIVES, baseline_particle)

        # Only the scratch particles with no corresponding baseline particle are left
        for scratch_particle in scratch_particles:
            nearest_in_baseline = find_nearest_particles(baseline_particles, around=scratch_particle, tolerance=1)
            distance_um = scratch_particle.distance_um(nearest_in_baseline[0], resolution)
            if distance_um > 3.3333 * max_distance_um:
                # Assume cell is in unannotated region
                continue
            report.add_data(_DETECTIONS_FALSE_POSITIVES, scratch_particle)

    return report
