from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets


def update_positions_for_changed_offsets(experiment: Experiment, previous_offsets: ImageOffsets):
    current_offsets = experiment.images.offsets
    for time_point in experiment.time_points():
        change_in_offset = current_offsets.of_time_point(time_point) - previous_offsets.of_time_point(time_point)
        if change_in_offset.is_zero():
            continue

        # Update the positions
        for position in set(experiment.positions.of_time_point(time_point)):
            #            ^ Make a copy so that collection is not modified while iterating over it
            experiment.move_position(position, position + change_in_offset, update_splines=False)

        # Update the splines
        for id, spline in experiment.splines.of_time_point(time_point):
            spline.move_points(change_in_offset)
