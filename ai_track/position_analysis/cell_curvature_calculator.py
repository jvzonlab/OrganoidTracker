"""A quick and dirty way to measure curvature of cells."""
# Find six nearest neighbors (N)
from typing import List, Tuple

from ai_track.core.experiment import Experiment
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection
from ai_track.core.resolution import ImageResolution
from ai_track.imaging import angles
from ai_track.imaging.lines import Line3, point_on_line_2_nearest_to_line_1
from ai_track.linking import nearby_position_finder


_NEARBY_CELLS_AMOUNT = 6


def get_curvature_angle(positions: PositionCollection, position_x: Position, resolution: ImageResolution):
    other_positions = positions.of_time_point(position_x.time_point())

    # Find six nearest neighbors (N)
    positions_around = nearby_position_finder.find_closest_n_positions(other_positions, around=position_x,
                                                                       max_amount=_NEARBY_CELLS_AMOUNT,
                                                                       resolution=resolution)
    if len(positions_around) < _NEARBY_CELLS_AMOUNT:
        return 0  # Not enough positions nearby to calculate

    vector_x = position_x.to_vector_um(resolution)
    found_angles = []

    # Take one neighbor, call it A
    for position_a in positions_around:

        # Find the point Q on the opposite site of X for A:    A --- X --- Q
        delta = position_x - position_a
        position_q = position_x + delta

        # Find the two nearest cells B and C from Q within (N), excluding A
        positions_around_without_current = positions_around.difference({position_a})
        position_b, position_c = nearby_position_finder.find_closest_n_positions(positions_around_without_current,
                                                                            around=position_q, resolution=resolution,
                                                                            max_amount=2)

        # Draw a line BC. This form a skew line with AQ. Find the nearest point R on line BC towards line AQ.
        vector_a = position_a.to_vector_um(resolution)
        vector_q = position_q.to_vector_um(resolution)
        vector_b = position_b.to_vector_um(resolution)
        vector_c = position_c.to_vector_um(resolution)
        line_aq = Line3.from_points(vector_a, vector_q)
        line_bc = Line3.from_points(vector_b, vector_c)
        vector_r = point_on_line_2_nearest_to_line_1(line_1=line_aq, line_2=line_bc)
        if vector_r == vector_x:
            # Very rare, but possible. Consider this to be a flat surface
            return 180

        # Calculate angle AXR
        angle = angles.right_hand_rule(vector_a, vector_x, vector_r)
        found_angles.append(angle)

    # Return average angle
    return sum(found_angles) / len(found_angles)
