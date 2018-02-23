"""Images and positions of particles (biological cells in our case)"""
from typing import List, Iterable, Optional, Dict
from networkx import Graph
from imaging import image_cache
from numpy import ndarray


class Particle:

    x: float
    y: float
    z: float
    _time_point_number: Optional[int]

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self._time_point_number = None

    def distance_squared(self, other) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * 5) ** 2

    def time_point_number(self):
        return self._time_point_number

    def with_time_point_number(self, time_point_number: int):
        if self._time_point_number is not None:
            raise ValueError("time_point_number was already set")
        self._time_point_number = int(time_point_number)
        return self

    def __repr__(self):
        string = "Particle(" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.0f" % self.z) + ")"
        if self._time_point_number is not None:
            string += ".with_time_point_number(" + str(self._time_point_number) + ")"
        return string

    def __str__(self):
        string = "cell at (" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.0f" % self.z) + ")"
        if self._time_point_number is not None:
            string += " at time point " + str(self._time_point_number)
        return string

    def __hash__(self):
        return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z)) ^ hash(int(self._time_point_number))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and abs(self.x - other.x) < 0.00001 and abs(self.x - other.x) < 0.00001 and abs(self.z - other.z) < 0.00001 \
               and self._time_point_number == other._time_point_number


class TimePoint:
    """A single point in time."""

    _time_point_number: int
    _particles: List[Particle]

    def __init__(self, time_point_number: int):
        self._time_point_number = time_point_number
        self._particles = []
        self._image_loader = None

    def time_point_number(self) -> int:
        return self._time_point_number

    def particles(self) -> List[Particle]:
        return self._particles

    def add_particles(self, particles: Iterable[Particle]) -> None:
        """Adds all particles in the list to this time_point. Throws ValueError if the particles were already assigned to
        a time_point."""
        for particle in particles:
            particle.with_time_point_number(self._time_point_number)
            self._particles.append(particle)

    def set_image_loader(self, loader):
        """Sets the image loader. The image loader must ba a function with no args, that returns a numpy
        multidimensional array. Each element in the array is another array that forms an image.
        """
        self._image_loader = loader

    def load_images(self, allow_cache=True) -> ndarray:
        if allow_cache:
            images = image_cache.get_from_cache(self._time_point_number)
            if images is not None:
                return images

        # Cache miss
        images = self._load_images_uncached()
        if allow_cache:
            image_cache.add_to_cache(self._time_point_number, images)
        return images

    def _load_images_uncached(self):
        image_loader = self._image_loader
        if self._image_loader is None:
            return None
        return image_loader()



class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class records the images, particle
     positions and particle trajectories."""

    _time_points: Dict[str, TimePoint]
    _particle_links: Optional[Graph]
    _particle_links_baseline: Optional[Graph] # Links that are assumed to be correct
    _first_time_point_number: Optional[int]
    _last_time_point_number: Optional[int]

    def __init__(self):
        self._time_points = {}
        self._particle_links = None
        self._particle_links_baseline = None
        self._last_time_point_number = None
        self._first_time_point_number = None

    def add_particles(self, time_point_number: int, raw_particles) -> None:
        """Adds particles to a time_point."""
        particles = []
        for raw_particle in raw_particles:
            particles.append(Particle(raw_particle[0], raw_particle[1], raw_particle[2]))
        time_point = self._get_or_add_time_point(time_point_number)
        time_point.add_particles(particles)

    def add_image_loader(self, time_point_number: int, image_loader) -> None:
        time_point = self._get_or_add_time_point(time_point_number)
        time_point.set_image_loader(image_loader)

    def get_time_point(self, time_point_number: int) -> TimePoint:
        """Gets the time_point with the given number. Throws KeyError if no such time_point exists."""
        return self._time_points[str(time_point_number)]

    def _get_or_add_time_point(self, time_point_number: int) -> TimePoint:
        try:
            return self._time_points[str(time_point_number)]
        except KeyError:
            time_point = TimePoint(time_point_number)
            self._time_points[str(time_point_number)] = time_point
            self._update_time_point_statistics(time_point_number)
            return time_point

    def _update_time_point_statistics(self, new_time_point_number: int):
        if self._first_time_point_number is None or self._first_time_point_number > new_time_point_number:
            self._first_time_point_number = new_time_point_number
        if self._last_time_point_number is None or self._last_time_point_number < new_time_point_number:
            self._last_time_point_number = new_time_point_number

    def first_time_point_number(self):
        if self._first_time_point_number is None:
            raise ValueError("No time_points exist")
        return self._first_time_point_number

    def last_time_point_number(self):
        if self._last_time_point_number is None:
            raise ValueError("No time_points exist")
        return self._last_time_point_number

    def get_previous_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time_point directly before the given time_point, or KeyError if the given time_point is the first time_point."""
        return self.get_time_point(time_point.time_point_number() - 1)

    def get_next_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time_point directly after the given time_point, or KeyError if the given time_point is the last time_point."""
        return self.get_time_point(time_point.time_point_number() + 1)

    def particle_links_scratch(self, network: Optional[Graph] = None) -> Optional[Graph]:
        """Gets or sets the particle linking results. It is not possible to replace exising results."""
        if network is not None:
            self._particle_links = network
        return self._particle_links

    def particle_links(self, network: Optional[Graph] = None) -> Optional[Graph]:
        """Gets or sets a particle linking result **that is known to be correct**."""
        if network is not None:
            self._particle_links_baseline = network
        return self._particle_links_baseline


def get_closest_particle(particles: Iterable[Particle], search_position: Particle,
                         ignore_z: bool = False, max_distance: int = 100000) -> Optional[Particle]:
    """Gets the particle closest ot the given position."""
    closest_particle = None
    closest_distance_squared = max_distance ** 2

    for particle in particles:
        if ignore_z:
            search_position.z = particle.z # Make search ignore z
        distance = particle.distance_squared(search_position)
        if distance < closest_distance_squared:
            closest_distance_squared = distance
            closest_particle = particle

    return closest_particle