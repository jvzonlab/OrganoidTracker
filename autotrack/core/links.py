from pprint import pprint
from typing import Optional, Dict, Iterable, List, Set, Union, Tuple, Any

from autotrack.core.particles import Particle


class LinkingTrack:
    _min_time_point_number: int  # Equal to _particles_by_time_point[0].time_point_number()

    # Particles by time point. Position 0 contains the particle at min_time_point, position 1 min_time_point + 1, etc.
    # List may contain None, but never as the first or last entry (then you would just resize the list)
    _particles_by_time_point: List[Optional[Particle]]
    _next_tracks: List["LinkingTrack"]
    _previous_tracks: List["LinkingTrack"]

    def __init__(self, particles_by_time_point: List[Particle]):
        self._min_time_point_number = particles_by_time_point[0].time_point_number()
        self._particles_by_time_point = particles_by_time_point
        self._next_tracks = list()
        self._previous_tracks = list()

    def _get_by_time_point(self, time_point_number: int):
        if time_point_number < self._min_time_point_number \
                or time_point_number >= self._min_time_point_number + len(self._particles_by_time_point):
            raise IndexError(f"Time point {time_point_number} outside track")
        return self._particles_by_time_point[time_point_number - self._min_time_point_number]

    def _find_pasts(self, time_point_number: int) -> Set[Particle]:
        """Returns all particles directly linked to the particle at the given time point."""
        search_index = (time_point_number - 1) - self._min_time_point_number # -1 is to look one time point in the past
        while search_index >= 0:
            previous_particle = self._particles_by_time_point[search_index]
            if previous_particle is not None:
                return {previous_particle}
            search_index -= 1

        # We ended up at the first time point of this track, continue search in previous tracks
        return {track.find_last() for track in self._previous_tracks}

    def _find_futures(self, time_point_number: int) -> Set[Particle]:
        """Returns all particles directly linked to the particle at the given time point."""
        search_index = (time_point_number + 1) - self._min_time_point_number
        while search_index < len(self._particles_by_time_point):
            next_particle = self._particles_by_time_point[search_index]
            if next_particle is not None:
                return {next_particle}
            search_index += 1

        # We ended up at the last time point of this track, continue search in next tracks
        return {track.find_first() for track in self._next_tracks}

    def find_first(self) -> Particle:
        """Returns the first particle in this track."""
        return self._particles_by_time_point[0]

    def find_last(self) -> Particle:
        """Returns the last particle in this track."""
        return self._particles_by_time_point[-1]

    def particles(self) -> Iterable[Particle]:
        """Returns all particles in this track, in order."""
        for particle in self._particles_by_time_point:
            if particle is not None:
                yield particle

    def _update_link_to_previous(self, was: "LinkingTrack", will_be: "LinkingTrack"):
        """Replaces a value in the _previous_tracks list. Make sure that old track is in the list."""
        self._previous_tracks.remove(was)
        self._previous_tracks.append(will_be)

    def _update_link_to_next(self, was: "LinkingTrack", will_be: "LinkingTrack"):
        """Replaces a value in the _next_tracks list. Make sure that old track is in the list."""
        self._next_tracks.remove(was)
        self._next_tracks.append(will_be)

    def max_time_point_number(self) -> int:
        """Gets the highest time point number where this track still contains a particle ."""
        return self._min_time_point_number + len(self._particles_by_time_point) - 1

    def get_age(self, particle: Particle) -> int:
        """Gets the age of this particle. This will be 0 if its the first track position, 1 on the position after that,
        etcetera."""
        return particle.time_point_number() - self._min_time_point_number

    def __repr__(self):
        return f"<LinkingTrack t={self._min_time_point_number}-{self.max_time_point_number()}>"

    def get_next_tracks(self) -> Set["LinkingTrack"]:
        """Gets a set of the tracks that will follow this track. If empty, the lineage end. If the length is 2, it is
        a cell division. Lengths of 1 will never occur. Lengths of 3 can occur, but make no biological sense."""
        return set(self._next_tracks)

    def get_previous_tracks(self) -> Set["LinkingTrack"]:
        """Gets a set of the tracks before this track. If empty, the lineage started. Normally, the set will have a
        size of 1. Larger sizes indicate a cell merge, which makes no biological sense."""
        return set(self._previous_tracks)


    def min_time_point_number(self) -> int:
        """Gets the first time point number of this track."""
        return self._min_time_point_number

    def __len__(self):
        """Gets the time length of the track, in number of time points."""
        return len(self._particles_by_time_point)


class ParticleLinks:
    """Represents all particle links. Two different networks can be specified (called baseline and scratch), so that
    comparisons become possible. Care has been taken to ensure that the node sets of both linking networks are
    equal, so that comparisons between the networks are easier."""

    _tracks: List[LinkingTrack]
    _particle_to_track: Dict[Particle, LinkingTrack]
    _particle_data: Dict[str, Dict[Particle, Any]]

    def __init__(self):
        self._tracks = []
        self._particle_to_track = dict()
        self._particle_data = dict()

    def add_links(self, links: "ParticleLinks"):
        """Adds all links from the graph. Existing link are not removed. Changes may write through in the original
        links."""
        if self.has_links():
            # Merge data
            for data_name, values in links._particle_data.items():
                if data_name not in self._particle_data:
                    self._particle_data[data_name] = values
                else:
                    self._particle_data[data_name].update(values)
            # Merge links
            for particle1, particle2 in links.find_all_links():
                self.add_link(particle1, particle2)
        else:
            self._tracks = links._tracks
            self._particle_to_track = links._particle_to_track
            self._particle_data = links._particle_data

    def remove_all_links(self):
        """Removes all links in the experiment."""
        for track in self._tracks:  # Help the garbage collector by removing all the cyclic dependencies
            track._next_tracks.clear()
            track._previous_tracks.clear()
        self._tracks.clear()
        self._particle_to_track.clear()

    def remove_links_of_particle(self, particle: Particle):
        """Removes all links from and to the particle."""
        track = self._particle_to_track.get(particle)
        if track is None:
            return

        age = track.get_age(particle)
        if len(track._particles_by_time_point) == 1:
            # This was the only particle in the track, remove track
            for previous_track in track._previous_tracks:
                previous_track._next_tracks.remove(track)
            for next_track in track._next_tracks:
                next_track._previous_tracks.remove(track)
            self._tracks.remove(track)
        elif age == 0:
            # Particle is first particle of the track
            # Remove links with previous tracks
            for previous_track in track._previous_tracks:
                previous_track._next_tracks.remove(track)
            track._previous_tracks = []

            # Remove actual particle
            track._particles_by_time_point[0] = None
            while track._particles_by_time_point[0] is None:  # Remove all Nones at the beginning
                track._min_time_point_number += 1
                track._particles_by_time_point = track._particles_by_time_point[1:]
        else:
            # Particle is further in the track
            if particle.time_point_number() < track.max_time_point_number():
                # Need to split so that particle is the last particle of the track
                _ = self._split_track(track, age + 1)

            # Decouple from next tracks
            for next_track in track._next_tracks:
                next_track._previous_tracks.remove(track)
            track._next_tracks = []

            # Delete last particle in the track
            track._particles_by_time_point[-1] = None
            while track._particles_by_time_point[-1] is None:  # Remove all Nones at the end
                del track._particles_by_time_point[-1]

        # Remove from indexes
        del self._particle_to_track[particle]
        for data_set in self._particle_data.values():
            if particle in data_set:
                del data_set[particle]

    def replace_particle(self, old_position: Particle, position_new: Particle):
        """Replaces one particle with another. The old particle is removed from the graph, the new one is added. All
        links will be moved over to the new particle"""
        if old_position.time_point_number() != position_new.time_point_number():
            raise ValueError("Cannot replace with particle at another time point")

        track = self._particle_to_track.get(old_position)
        if track is None:
            return

        track._particles_by_time_point[position_new.time_point_number() - track._min_time_point_number] = position_new
        del self._particle_to_track[old_position]
        self._particle_to_track[position_new] = track

    def has_links(self) -> bool:
        """Returns True if the graph is not None."""
        return len(self._particle_to_track) > 0

    def to_d3_data(self) -> Dict:
        """Return data in D3.js node-link format that is suitable for JSON serialization
        and use in Javascript documents."""
        nodes = list()

        # Save nodes and store extra data
        for particle in self.find_all_particles():
            node = {
                "id": particle
            }
            for data_name, data_values in self._particle_data.items():
                particle_value = data_values.get(particle)
                if particle_value is not None:
                    node[data_name] = particle_value
            nodes.append(node)

        # Save edges
        edges = list()
        for source, target in self.find_all_links():
            edge = {
                "source": source,
                "target": target
            }
            edges.append(edge)

        return {
            "directed": False,
            "multigraph": False,
            "graph": dict(),
            "nodes": nodes,
            "links": edges
        }

    def add_d3_data(self, data: Dict, min_time_point: int = -100000, max_time_point: int = 100000):
        """Adds data in the D3.js node-link format. Used for deserialization."""

        # Add particle data
        for node in data["nodes"]:
            if len(node.keys()) == 1:
                # No extra data found
                continue
            particle = node["id"]
            for data_key, data_value in node.items():
                if data_key == "id":
                    continue
                self.set_particle_data(particle, data_key, data_value)

        # Add links
        for link in data["links"]:
            source: Particle = link["source"]
            target: Particle = link["target"]
            if source.time_point_number() < min_time_point or target.time_point_number() < min_time_point \
                or source.time_point_number() > max_time_point or target.time_point_number() > max_time_point:
                continue
            self.add_link(source, target)

    def find_futures(self, particle: Particle) -> Set[Particle]:
        """Returns all connections to the future."""
        track = self._particle_to_track.get(particle)
        if track is None:
            return set()
        return track._find_futures(particle.time_point_number())

    def find_pasts(self, particle: Particle) -> Set[Particle]:
        """Returns all connections to the past."""
        track = self._particle_to_track.get(particle)
        if track is None:
            return set()
        return track._find_pasts(particle.time_point_number())

    def find_appeared_cells(self, time_point_number_to_ignore: Optional[int] = None) -> Iterable[Particle]:
        """This method gets all particles that "popped up out of nothing": that have no links to the past. You can give
        this method a time point number to ignore. Usually, this would be the first time point number of the experiment,
        as cells that have no links to the past in the first time point are not that interesting."""
        for track in self._tracks:
            if time_point_number_to_ignore is None or time_point_number_to_ignore != track._min_time_point_number:
                yield track.find_first()

    def add_link(self, particle1: Particle, particle2: Particle):
        """Adds a link between the particles. The linking network will be initialized if necessary."""
        if particle1.time_point_number() == particle2.time_point_number():
            raise ValueError("Particles are in the same time point")

        track1 = self._particle_to_track.get(particle1)
        track2 = self._particle_to_track.get(particle2)

        if track1 is not None and track2 is not None and self.contains_link(particle1, particle2):
            return  # Already has that link, don't add a second link (this will corrupt the data structure)

        if track1 is not None and track2 is None:
            if track1.max_time_point_number() == particle1.time_point_number() \
                    and not track1._next_tracks \
                    and particle2.time_point_number() == particle1.time_point_number() + 1:
                # This very common case of adding a single particle to a track is singled out
                # It could be handled just fine by the code below, which will create a new track and then merge the
                # tracks, but this is faster
                track1._particles_by_time_point.append(particle2)
                self._particle_to_track[particle2] = track1
                return

        if track1 is None:  # Create new mini-track
            track1 = LinkingTrack([particle1])
            self._tracks.append(track1)
            self._particle_to_track[particle1] = track1

        if track2 is None:  # Create new mini-track
            track2 = LinkingTrack([particle2])
            self._tracks.append(track2)
            self._particle_to_track[particle2] = track2

        # Make sure particle1 comes first in time
        if particle1.time_point_number() > particle2.time_point_number():
            track1, track2 = track2, track1  # Switch around to make particle1 always the earliest
            particle1, particle2 = particle2, particle1

        if particle1.time_point_number() < track1.max_time_point_number():
            # Need to split track 1 so that particle1 is at the end
            _ = self._split_track(track1, track1.get_age(particle1) + 1)

        if particle2.time_point_number() > track2._min_time_point_number:
            # Need to split track 2 so that particle2 is at the start
            part_after_split = self._split_track(track2, track2.get_age(particle2))
            track2 = part_after_split

        # Connect the tracks
        track1._next_tracks.append(track2)
        track2._previous_tracks.append(track1)
        self._try_merge(track1, track2)

    def get_particle_data(self, particle: Particle, data_name: str) -> Union[str, int, None]:
        """Gets the attribute of the particle with the given name. Returns None if not found."""
        data_of_particles = self._particle_data.get(data_name)
        if data_of_particles is None:
            return None
        return data_of_particles.get(particle)

    def set_particle_data(self, particle: Particle, data_name: str, value: Union[str, int, None]):
        """Adds or overwrites the given attribute for the given particle. Set value to None to delete the attribute.

        Note: this is a low-level API. See the linking_markers module for more high-level methods, for example for how
        to read end markers, error markers, etc.
        """
        if data_name == "id":
            raise ValueError("The data_name 'id' is used to store the particle itself.")
        data_of_particles = self._particle_data.get(data_name)
        if data_of_particles is None:
            data_of_particles = dict()
            self._particle_data[data_name] = data_of_particles
        data_of_particles[particle] = value

    def find_links_of(self, particle: Particle) -> Iterable[Particle]:
        """Gets all links of a particle, both to the past and the future."""
        track = self._particle_to_track.get(particle)
        if track is None:
            return set()
        return track._find_futures(particle.time_point_number()) | track._find_pasts(particle.time_point_number())

    def find_all_particles(self) -> Iterable[Particle]:
        """Gets all particles in the linking graph. Note that particles without links are not included here."""
        return self._particle_to_track.keys()

    def remove_link(self, particle1: Particle, particle2: Particle):
        """Removes the link between the given particles. Does nothing if there is no link between the particles."""
        if particle1.time_point_number() > particle2.time_point_number():
            particle2, particle1 = particle1, particle2

        if particle1.time_point_number() == particle2.time_point_number():
            return  # No link can possibly exist

        track1 = self._particle_to_track.get(particle1)
        track2 = self._particle_to_track.get(particle2)
        if track1 is None or track2 is None:
            return  # No link exists
        if track1 == track2:
            # So particles are in the same track

            # Check if there is nothing in between
            for time_point_number in range(particle1.time_point_number() + 1, particle2.time_point_number()):
                if track1._get_by_time_point(time_point_number) is not None:
                    return  # There's a particle in between, so the specified link doesn't exist

            # Split directly after particle1
            new_track = self._split_track(track1, particle1.time_point_number() + 1 - track1._min_time_point_number)
            track1._next_tracks = []
            new_track._previous_tracks = []
        else:
            # Check if the tracks connect
            if not track1.max_time_point_number() == particle1.time_point_number():
                # Particle 1 is not the last particle in its track, so it cannot be connected to another track
                return
            if not track2._min_time_point_number == particle2.time_point_number():
                # Particle 2 is not the first particle in its track, so it cannot be connected to another track
                return

            # The tracks may be connected. Remove the connection, if any
            if track2 in track1._next_tracks:
                track1._next_tracks.remove(track2)
                track2._previous_tracks.remove(track1)

    def contains_link(self, particle1: Particle, particle2: Particle) -> bool:
        """Returns True if the two given particles are linked to each other."""
        if particle1.time_point_number() == particle2.time_point_number():
            return False  # Impossible to have a link

        if particle1.time_point_number() < particle2.time_point_number():
            return particle1 in self.find_pasts(particle2)
        else:
            return particle1 in self.find_futures(particle2)

    def contains_particle(self, particle: Particle) -> bool:
        """Returns True if the given particle is part of this linking network."""
        return particle in self._particle_to_track

    def find_all_links(self) -> Iterable[Tuple[Particle, Particle]]:
        """Gets all available links."""
        for track in self._tracks:
            # Return inter-track links
            previous_particle = None
            for particle in track.particles():
                if previous_particle is not None:
                    yield previous_particle, particle
                previous_particle = particle

            # Return links to next track
            for next_track in track._next_tracks:
                yield previous_particle, next_track.find_first()

            # (links to previous track are NOT returned, those will be included by that previous track as links to the
            # next track)

    def copy(self) -> "ParticleLinks":
        """Returns a copy of all the links, so that you can modify that data set without affecting this one."""
        copy = ParticleLinks()

        # Copy over tracks
        for track in self._tracks:
            copied_track = LinkingTrack(track._particles_by_time_point.copy())
            copy._tracks.append(copied_track)
            for particle in track.particles():
                copy._particle_to_track[particle] = copied_track

        # We can now re-establish the links between all tracks
        for track in self._tracks:
            track_copy = copy._particle_to_track[track.find_first()]
            for next_track in track._next_tracks:
                next_track_copy = copy._particle_to_track[next_track.find_first()]
                track_copy._next_tracks.append(next_track_copy)
                next_track_copy._previous_tracks.append(track_copy)

        # Don't forget to copy over data
        for data_name, data_value in self._particle_data.items():
            copy._particle_data[data_name] = data_value.copy()

        return copy

    def _split_track(self, old_track: LinkingTrack, split_index: int) -> LinkingTrack:
        """Modifies the given track so that all particles after a certain time points are removed, and placed in a new
        track. So particles[0:split_index] will remain in this track, particles[split_index:] will be moved."""
        particles_after_split = old_track._particles_by_time_point[split_index:]

        # Remove None at front (this is safe, as the last particle in the track may never be None)
        while particles_after_split[0] is None:
            particles_after_split = particles_after_split[1:]

        # Delete particles from after the split from original track
        del old_track._particles_by_time_point[split_index:]

        # Create a new track, add all connections
        track_after_split = LinkingTrack(particles_after_split)
        track_after_split._next_tracks = old_track._next_tracks
        for new_next_track in track_after_split._next_tracks:
            new_next_track._update_link_to_previous(was=old_track, will_be=track_after_split)
        track_after_split._previous_tracks = [old_track]
        old_track._next_tracks = [track_after_split]

        # Update indices for changed tracks
        self._tracks.append(track_after_split)
        for particle_after_split in particles_after_split:
            self._particle_to_track[particle_after_split] = track_after_split

        return track_after_split

    def _try_merge(self, track1: LinkingTrack, track2: LinkingTrack):
        """Call this method with two tracks that are ALREADY connected to each other. If possible, they will be merged
        into one big track."""
        if track1._min_time_point_number > track2._min_time_point_number:
            first_track, second_track = track2, track1
        else:
            first_track, second_track = track1, track2

        if len(second_track._previous_tracks) != 1 or len(first_track._next_tracks) != 1:
            return  # Cannot be merged
        # Ok, they can be merged into just first_track. Move all particles over to first_track.
        gap_length = second_track._min_time_point_number -\
                     (first_track._min_time_point_number + len(first_track._particles_by_time_point))
        if gap_length != 0:
            first_track._particles_by_time_point += [None] * gap_length
        first_track._particles_by_time_point += second_track._particles_by_time_point

        # Update registries
        self._tracks.remove(second_track)
        for moved_particle in second_track.particles():
            self._particle_to_track[moved_particle] = first_track
        first_track._next_tracks = second_track._next_tracks
        for new_next_track in first_track._next_tracks:  # Notify all next tracks that they have a new predecessor
            new_next_track._update_link_to_previous(second_track, first_track)

    def debug_sanity_check(self):
        """Checks if the data structure still has a valid structure. If not, this method throws ValueError. This should
        never happen if you only use the public methods (those without a _ at the start), and don't poke around in
        internal code.

        This method is very useful to debug the data structure if you get some weird results."""
        for particle, track in self._particle_to_track.items():
            if track not in self._tracks:
                raise ValueError(f"{track} is not in the track list, but is in the index for particle {particle}")

        for track in self._tracks:
            if len(track._particles_by_time_point) == 0:
                raise ValueError(f"Empty track at t={track._min_time_point_number}")
            if track.find_first() is None:
                raise ValueError(f"{track} has no first particle")
            if track.find_last() is None:
                raise ValueError(f"{track} has no last particle")
            for particle in track.particles():
                if particle not in self._particle_to_track:
                    raise ValueError(f"{particle} of {track} is not indexed")
                elif self._particle_to_track[particle] != track:
                    raise ValueError(f"{particle} in track {track} is indexed as being in track"
                                     f" {self._particle_to_track[particle]}")
            for previous_track in track._previous_tracks:
                if previous_track.max_time_point_number() >= track._min_time_point_number:
                    raise ValueError(f"Previous track {previous_track} is not in the past compared to {track}")
                if track not in previous_track._next_tracks:
                    raise ValueError(f"Current track {track} is connected to previous track {previous_track}, but that"
                                     f" track is not connected to the current track.")
            if len(track._next_tracks) == 1 and len(track._next_tracks[0]._previous_tracks) == 1:
                raise ValueError(f"Track {track} and {track._next_tracks[0]} could have been merged into"
                                 f" a single track")

    def find_starting_tracks(self) -> Iterable[LinkingTrack]:
        """Gets all starting tracks, so all tracks that have no links to the past."""
        for track in self._tracks:
            if not track._previous_tracks:
                yield track
