"""Finds all positions that move faster than the surrounding positions."""
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection
from ai_track.core.resolution import ImageResolution
from ai_track.linking import nearby_position_finder


_MAX_AMOUNT = 6


def _get_relative_movement(position: Position, all_positions: PositionCollection, links: Links,
                           resolution: ImageResolution) -> float:
    in_time_point = all_positions.of_time_point(position.time_point())
    nearby_positions = nearby_position_finder.find_closest_n_positions(in_time_point, around=position,
                                                                       max_amount=_MAX_AMOUNT, resolution=resolution)

