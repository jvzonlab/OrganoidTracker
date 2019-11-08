from ai_track.comparison.report import ComparisonReport, Category, Statistics
from ai_track.core.experiment import Experiment
from ai_track.linking.nearby_position_finder import find_close_positions, find_closest_position

_DETECTIONS_FALSE_NEGATIVES = Category("Missed detections")
_DETECTIONS_TRUE_POSITIVES = Category("Found detections")
_DETECTIONS_FALSE_POSITIVES = Category("Made up detections")
_DETECTIONS_REJECTED = Category("Rejected detections")


class DetectionReport(ComparisonReport):

    def __init__(self):
        super().__init__()
        self.title = "Detection comparison"

    def calculate_time_detection_statistics(self) -> Statistics:
        return self.calculate_time_statistics(_DETECTIONS_TRUE_POSITIVES, _DETECTIONS_FALSE_POSITIVES,
                                              _DETECTIONS_FALSE_NEGATIVES)

    def calculate_z_detection_statistics(self) -> Statistics:
        return self.calculate_z_statistics(_DETECTIONS_TRUE_POSITIVES, _DETECTIONS_FALSE_POSITIVES,
                                           _DETECTIONS_FALSE_NEGATIVES)


def compare_positions(ground_truth: Experiment, scratch: Experiment, max_distance_um: float = 5,
                      rejection_distance_um: float = 5) -> DetectionReport:
    """Checks how much the positions in the ground truth match with the given data."""
    resolution = ground_truth.images.resolution()

    report = DetectionReport()
    report.summary = f"Comparison of two sets of positions. The ground truth was named \"{ground_truth.name}\", the" \
                     f" comparision object was named \"{scratch.name}\"."

    for time_point in ground_truth.time_points():
        baseline_positions = set(ground_truth.positions.of_time_point(time_point))
        if len(baseline_positions) == 0:
            continue  # Nothing to compare for this time point

        scratch_positions = set(scratch.positions.of_time_point(time_point))
        for baseline_position in baseline_positions:
            nearest_in_scratch = find_close_positions(scratch_positions, around=baseline_position, tolerance=1,
                                                      resolution=resolution)
            if len(nearest_in_scratch) == 0:
                report.add_data(_DETECTIONS_FALSE_NEGATIVES, baseline_position, "No candidates in the scratch data left.")
                continue
            distance_um = baseline_position.distance_um(nearest_in_scratch[0], resolution)
            if distance_um > max_distance_um:
                report.add_data(_DETECTIONS_FALSE_NEGATIVES, baseline_position, f"Nearest cell was {distance_um:0.1f} um away")
                continue
            scratch_positions.remove(nearest_in_scratch[0])
            report.add_data(_DETECTIONS_TRUE_POSITIVES, baseline_position)

        # Only the scratch positions with no corresponding baseline position are left
        baseline_positions = set(ground_truth.positions.of_time_point(time_point))
        for scratch_position in scratch_positions:
            nearest_in_baseline = find_closest_position(baseline_positions, around=scratch_position,
                                                        resolution=resolution)
            distance_um = scratch_position.distance_um(nearest_in_baseline, resolution)
            if distance_um > rejection_distance_um:
                report.add_data(_DETECTIONS_REJECTED, scratch_position,
                                f"Nearest ground-truth cell was {distance_um:0.1f} um away")
                continue
            report.add_data(_DETECTIONS_FALSE_POSITIVES, scratch_position,
                            f"Nearest ground-truth cell was {distance_um:0.1f} um away")

    return report
