from typing import Dict, Optional, List, Tuple, Iterable

import networkx
from networkx import Graph
from numpy import ndarray

from autotrack.core import TimePoint, Name
from autotrack.core.image_loader import ImageLoader
from autotrack.core.links import ParticleLinks, LinkType
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.path import PathCollection
from autotrack.core.score import ScoreCollection


class _CachedImageLoader(ImageLoader):
    """Wrapper that caches the last few loaded images."""

    _internal: ImageLoader
    _image_cache: List[Tuple[int, ndarray]]

    def __init__(self, wrapped: ImageLoader):
        self._image_cache = []
        self._internal = wrapped

    def _add_to_cache(self, time_point_number: int, image: ndarray):
        if len(self._image_cache) > 5:
            self._image_cache.pop(0)
        self._image_cache.append((time_point_number, image))

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        for entry in self._image_cache:
            if entry[0] == time_point_number:
                return entry[1]

        # Cache miss
        image = self._internal.get_image_stack(time_point)
        self._add_to_cache(time_point_number, image)
        return image

    def get_resolution(self):
        return self._internal.get_resolution()

    def uncached(self) -> ImageLoader:
        return self._internal

    def get_first_time_point(self) -> Optional[int]:
        return self._internal.get_first_time_point()

    def get_last_time_point(self) -> Optional[int]:
        return self._internal.get_last_time_point()


class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class ultimately collects all
    details of the experiment."""

    # Note: none of the fields may be None after __init__ is called
    _particles: ParticleCollection
    scores: ScoreCollection
    _links: ParticleLinks
    _image_loader: ImageLoader = ImageLoader()
    _name: Name
    paths: PathCollection

    def __init__(self):
        self._name = Name()
        self._particles = ParticleCollection()
        self.scores = ScoreCollection()
        self.paths = PathCollection()
        self._links = ParticleLinks()

    def add_particle_raw(self, x: float, y: float, z: float, time_point_number: int):
        """Adds a single particle to the experiment, creating the time point if it does not exist yet."""
        particle = Particle(x, y, z)
        particle.with_time_point_number(time_point_number)
        self.add_particle(particle)

    def add_particle(self, particle: Particle):
        """Adds a particle to the experiment. The particle must have a time point number specified."""
        self._particles.add(particle)

    def remove_particle(self, particle: Particle):
        self._particles.detach_particle(particle)
        self._links.remove_links_of_particle(particle)

    def move_particle(self, old_position: Particle, position_new: Particle) -> bool:
        """Moves the position of a particle, preserving any links. (So it's different from remove-and-readd.) The shape
        of a particle is not preserved, though. Throws ValueError when the particle is moved to another time point. If
        the new position has not time point specified, it is set to the time point o the existing particle."""
        position_new.with_time_point_number(old_position.time_point_number())  # Make sure both have the same time point

        # Replace in linking graphs
        self._links.replace_particle(old_position, position_new)

        # Replace in particle collection
        self._particles.detach_particle(old_position)
        self._particles.add(position_new)
        return True

    def remove_particles(self, time_point: TimePoint):
        """Removes the particles and links of a given time point."""
        for particle in self._particles.of_time_point(time_point):
            self._links.remove_links_of_particle(particle)
        self._particles.detach_all_for_time_point(time_point)

    def remove_all_particles(self):
        """Removes all particles and links in the experiment, so in all time points."""
        self._particles = ParticleCollection()
        self._links.remove_all_links()
        self.scores = ScoreCollection()

    def get_time_point(self, time_point_number: int) -> TimePoint:
        """Gets the time point with the given number. Throws ValueError if no such time point exists."""
        first = self.first_time_point_number()
        last = self.last_time_point_number()
        if first is None or last is None:
            raise ValueError("No time points have been loaded yet")
        if time_point_number < first or time_point_number > last:
            raise ValueError(f"Time point out of bounds (was: {time_point_number}, first: {first}, last: {last})")
        return TimePoint(time_point_number)

    def first_time_point_number(self):
        """Gets the first time point of the experiment where there is data (images and/or particles)."""
        image_first = self._image_loader.get_first_time_point()
        particle_first = self._particles.first_time_point_number()
        if image_first is None:
            return particle_first
        if particle_first is None:
            return image_first
        return min(particle_first, image_first)

    def last_time_point_number(self):
        """Gets the last time point (inclusive) of the experiment where there is data (images and/or particles)."""
        image_last = self._image_loader.get_last_time_point()
        particle_last = self._particles.last_time_point_number()
        if image_last is None:
            return particle_last
        if particle_last is None:
            return image_last
        return max(particle_last, image_last)

    def get_previous_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly before the given time point. Throws KeyError if the given time point is the last
        time point."""
        return self.get_time_point(time_point.time_point_number() - 1)

    def get_next_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly after the given time point. Throws KeyError if the given time point is the last
         time point."""
        return self.get_time_point(time_point.time_point_number() + 1)

    def time_points(self) -> Iterable[TimePoint]:
        first_number = self.first_time_point_number()
        last_number = self.last_time_point_number()
        if first_number is None or last_number is None:
            return []

        current_number = first_number
        while current_number <= last_number:
            yield self.get_time_point(current_number)
            current_number += 1

    def image_loader(self, image_loader: Optional[ImageLoader] = None) -> ImageLoader:
        """Gets/sets the image loader."""
        if image_loader is not None:
            self._image_loader = _CachedImageLoader(image_loader.uncached())
            return image_loader
        return self._image_loader

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        """Gets a stack of all images for a time point, one for every z layer. Returns None if there is no image."""
        return self._image_loader.get_image_stack(time_point)

    @property
    def particles(self) -> ParticleCollection:
        """Gets all particles of all time points."""
        return self._particles

    @property
    def name(self) -> Name:
        # Don't allow to replace the Name object
        return self._name

    @property
    def links(self) -> ParticleLinks:
        """Gets all links between the particles of different time points."""
        # Don't allow to replace the ParticleLinks object
        return self._links

    @property
    def division_lookahead_time_points(self):
        """Where there no divisions found because a cell really didn't divide, or did the experiment simply end before
        the cell divided? If the experiment continues for at least this many time points, then we can safely assume that
         the cell did not divide."""
        return 100
