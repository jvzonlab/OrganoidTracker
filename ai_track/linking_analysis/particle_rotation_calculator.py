from typing import List, Tuple

from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.imaging import angles, lines
from ai_track.imaging.lines import Line3


def calculate_rotation_of_track(links: Links, resolution: ImageResolution, starting_position: Position, starting_center_p1: Position,
                                starting_center_p2: Position, ending_center_p1: Position) -> Tuple[List[float], List[Position]]:
    """Calculates the cumulative rotation of the position relative to the center. Also returns the final position. Both
    the position and the center must have links towards the future. The rotation is calculated for as long as the center
    has links to the future, or until the max_time_point_number has been reached. This method returns multiple values,
    to support dividing cells. If a cell divides, and both daughters also divide, then you end up with four values.
    Particles that disappear before the max time point is reached, are not included in the result."""
    if starting_position.time_point_number() != starting_center_p1.time_point_number():
        raise ValueError("Starting position and starting center must be in the same time point")
    if starting_center_p1.time_point_number() != starting_center_p2.time_point_number():
        raise ValueError("Starting center points not in the same time point")
    if ending_center_p1.time_point_number() == starting_center_p1.time_point_number():
        return [0], [starting_position]  # Zero time points, so zero rotation
    if ending_center_p1.time_point_number() < starting_center_p1.time_point_number():
        raise ValueError("Ending position is earlier than starting position")

    current_position = starting_position
    duration = ending_center_p1.time_point_number() - starting_center_p1.time_point_number()

    cumulative_rotation = 0
    rotation_axis = Line3(starting_center_p1.to_vector_um(resolution), starting_center_p2.to_vector_um(resolution))
    current_orientation = lines.direction_to_point(rotation_axis, starting_position.to_vector_um(resolution))

    while True:
        # Loop over time
        time_fraction_next = (current_position.time_point_number() - starting_position.time_point_number() + 1) / duration

        # Use linear interpolation to find the next center
        next_center_p1 = Position(time_fraction_next * starting_center_p1.x + (1 - time_fraction_next) * ending_center_p1.x,
                                  time_fraction_next * starting_center_p1.y + (1 - time_fraction_next) * ending_center_p1.y,
                                  time_fraction_next * starting_center_p1.z + (1 - time_fraction_next) * ending_center_p1.z,
                                  time_point_number=current_position.time_point_number() + 1)

        # Check if we've gone too far in time
        if next_center_p1.time_point_number() > ending_center_p1.time_point_number():
            # End of center track - stop following cells
            return [cumulative_rotation], [current_position]

        next_positions = links.find_futures(current_position)
        if len(next_positions) == 0:
            # We don't know the rotation - premature disappearance of cell
            return [], []

        # Need to calculate rotation axis for this time point
        next_center_p2 = next_center_p1 + starting_center_p2 - starting_center_p1
        rotation_axis = Line3(next_center_p1.to_vector_um(resolution), next_center_p2.to_vector_um(resolution))

        if len(next_positions) > 1:
            # Division - return all resulting values
            return_rotations = []
            return_positions = []
            for next_position in next_positions:
                # Calculate rotation during division
                next_orientation = lines.direction_to_point(rotation_axis, next_position.to_vector_um(resolution))
                rotation_during_division = angles.difference(current_orientation, next_orientation)

                # Calculate rotation of daughter track (including the daughters of the daugthers, etc.)
                daughter_cumulative_rotations, daughter_final_positions = calculate_rotation_of_track(
                    links, resolution, next_position, next_center_p1, next_center_p2, ending_center_p1)

                # Add up all rotations together: everything up to now + rotation during division + everything afterwards
                for daughter_cumulative_rotation, daughter_final_position in zip(daughter_cumulative_rotations,
                                                                                 daughter_final_positions):
                    return_rotations.append(cumulative_rotation + rotation_during_division + daughter_cumulative_rotation)
                    return_positions.append(daughter_final_position)
            return return_rotations, return_positions

        # Here len(next_positions) == 1

        # Calculate rotation
        next_position = next_positions.pop()
        next_orientation = lines.direction_to_point(rotation_axis, next_position.to_vector_um(resolution))
        cumulative_rotation += angles.difference(current_orientation, next_orientation)

        # Advance to next loop
        current_orientation = next_orientation
        current_position = next_position
