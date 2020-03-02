"""Finds all positions that move faster than the surrounding positions."""
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.linking import nearby_position_finder


_MAX_AMOUNT = 6


def _get_relative_movement(position: Position, all_positions: PositionCollection, links: Links,
                           resolution: ImageResolution) -> float:
    in_time_point = all_positions.of_time_point(position.time_point())
    nearby_positions = nearby_position_finder.find_closest_n_positions(in_time_point, around=position,
                                                                       max_amount=_MAX_AMOUNT, resolution=resolution)

