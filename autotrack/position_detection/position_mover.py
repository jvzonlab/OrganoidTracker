from autotrack.core.experiment import Experiment
from autotrack.core.images import ImageOffsets


def update_positions_for_changed_offsets(experiment: Experiment, previous_offsets: ImageOffsets):
    current_offsets = experiment.images.offsets
    for time_point in experiment.time_points():
        change_in_offset = current_offsets.of_time_point(time_point).subtract_pos(
            previous_offsets.of_time_point(time_point))
        if change_in_offset.is_zero():
            continue
        for position in set(experiment.positions.of_time_point(time_point)):
            #            ^ Make a copy so that collection is not modified while iterating over it
            experiment.move_position(position, position.add_pos(change_in_offset))
        for id, data_axis in experiment.data_axes.of_time_point(time_point):
            data_axis.move_points(change_in_offset)
