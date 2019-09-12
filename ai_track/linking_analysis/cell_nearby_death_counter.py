"""Goes back in time, and records how many times a nearby cell has died. For create an instance of NearbyDeaths (which
sets up an index to speed up later calculations), then call its method count_nearby_deaths_in_past."""

from typing import Dict, List

from ai_track.core.links import Links, LinkingTrack
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection
from ai_track.core.resolution import ImageResolution
from ai_track.linking_analysis import linking_markers


class NearbyDeaths:
    """Goes back in time, and records how many times a nearby (within max_distance_um) cell has died."""

    # Two indexes, to speed up the calculations
    _track_to_nearby_deaths: Dict[LinkingTrack, List[Position]]
    _track_to_previous_death_counts: Dict[LinkingTrack, int]

    def __init__(self, links: Links, resolution: ImageResolution, max_distance_um: float = 18):
        """Does the initial bookkeeping, to speed up later calculations."""
        death_positions = PositionCollection(linking_markers.find_death_and_shed_positions(links))
        max_distance_squared = max_distance_um ** 2

        # List all deaths nearby each track
        self._track_to_nearby_deaths = dict()
        for track in links.find_all_tracks():
            nearby_deaths = []
            for position in track.positions():
                for death_position in death_positions.of_time_point(position.time_point()):
                    if death_position.distance_squared(position, resolution) > max_distance_squared:
                        continue

                    nearby_deaths.append(death_position)
            self._track_to_nearby_deaths[track] = nearby_deaths

        # For faster lookup, calculate how many deaths there were in previous tracks of that track
        self._track_to_previous_death_counts = dict()
        for track in links.find_all_tracks():
            previous_death_counts = 0

            previous_tracks = track.get_previous_tracks()
            while len(previous_tracks) == 1:
                previous_track = previous_tracks.pop()
                previous_death_counts += len(self._track_to_nearby_deaths[previous_track])
                previous_tracks = previous_track.get_previous_tracks()

            self._track_to_previous_death_counts[track] = previous_death_counts

    def count_nearby_deaths_in_past(self, links: Links, position: Position) -> int:
        """Calculates the number of deaths in the past."""
        track = links.get_track(position)
        if track is None:
            return 0

        total_count = 0

        deaths_nearby_track = self._track_to_nearby_deaths.get(track)
        if deaths_nearby_track is None:
            deaths_nearby_track = []
        for death_nearby_track in deaths_nearby_track:
            if death_nearby_track.time_point_number() <= position.time_point_number():
                total_count += 1

        death_count_before_track = self._track_to_previous_death_counts.get(track)
        if death_count_before_track is not None:
            total_count += death_count_before_track

        return total_count
