from ai_track.comparison.report import ComparisonReport, Category, Statistics
from ai_track.core.experiment import Experiment
from ai_track.linking.nearby_position_finder import find_close_positions


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
    resolution = ground_truth.images.resolution()

    report = DetectionReport()

    for time_point in ground_truth.time_points():
        baseline_positions = set(ground_truth.positions.of_time_point(time_point))
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
        for scratch_position in scratch_positions:
            nearest_in_baseline = find_close_positions(baseline_positions, around=scratch_position, tolerance=1,
                                                       resolution=resolution)
            distance_um = scratch_position.distance_um(nearest_in_baseline[0], resolution)
            if distance_um > 3.3333 * max_distance_um:
                # Assume cell is in unannotated region
                continue
            report.add_data(_DETECTIONS_FALSE_POSITIVES, scratch_position)

    return report
