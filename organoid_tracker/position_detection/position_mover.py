from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.core.position import Position


def update_positions_for_changed_offsets(experiment: Experiment, previous_offsets: ImageOffsets):
    current_offsets = experiment.images.offsets
    for time_point in experiment.time_points():
        change_in_offset = current_offsets.of_time_point(time_point) - previous_offsets.of_time_point(time_point)
        if change_in_offset.is_zero():
            continue

        # Update the positions
        old_positions = set(experiment.positions.of_time_point(time_point))
        for old_position in old_positions:
            #            ^ Make a copy so that collection is not modified while iterating over it
            new_position = old_position + change_in_offset
            while new_position in old_positions:
                # Position A moves to a location that is already occupied by an existing position B.
                # Adjust the new position slightly so that it is not the same as the existing position B.
                new_position = new_position + Position(0.01, 0.01, 0)
            experiment.move_position(old_position, new_position, update_splines=False)

        # Update the splines
        for id, spline in experiment.splines.of_time_point(time_point):
            spline.move_points(change_in_offset)
