"""Some base classes. Quick overview: Particles (usually cells, but may also be artifacts) are placed in TimePoints,
which are placed in an Experiment. A TimePoint also stores scores of possible mother-daughter cell combinations.
An Experiment also stores an ImageLoader and up to two cell links networks (stored as Graph objects)."""
from operator import itemgetter
from typing import List, Iterable, Optional, Dict, Set, Any, Union, AbstractSet

from networkx import Graph
from numpy import ndarray

from core.shape import ParticleShape, UnknownShape

COLOR_CELL_NEXT = "red"
COLOR_CELL_PREVIOUS = "blue"
COLOR_CELL_CURRENT = "lime"


class Particle:
    """A detected particle. Only the 3D + time position is stored here, see the ParticleShape class for the shape."""
    x: float
    y: float
    z: float
    _time_point_number: Optional[int] = None

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def distance_squared(self, other: "Particle", z_factor: float = 5) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * z_factor) ** 2

    def time_point_number(self):
        return self._time_point_number

    def with_time_point_number(self, time_point_number: int):
        if self._time_point_number is not None and self._time_point_number != time_point_number:
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


class Score:
    """Represents some abstract score, calculated from the individual elements. Usage:

        score = Score()
        score.foo = 4
        score.bar = 3.1
        # Results in score.total() == 7.1
    """

    def __init__(self, **kwargs):
        self.__dict__["scores"] = kwargs.copy()

    def __setattr__(self, key, value):
        self.__dict__["scores"][key] = value

    def __getattr__(self, item):
        return self.__dict__["scores"][item]

    def __delattr__(self, item):
        del self.__dict__["scores"][item]

    def total(self):
        score = 0
        for name, value in self.__dict__["scores"].items():
            score += value
        return score

    def keys(self) -> List[str]:
        keylist = list(self.__dict__["scores"].keys())
        keylist.sort()
        return keylist

    def get(self, key: str) -> float:
        """Gets the specified score, or 0 if it does not exist"""
        try:
            return self.__dict__["scores"][key]
        except KeyError:
            return 0.0

    def dict(self) -> Dict[str, float]:
        """Gets the underlying score dictionary"""
        return self.__dict__["scores"]

    def is_likely_mother(self):
        """Uses a simple threshold to check whether it is likely that this mother is a mother cell."""
        return self.total() > 3

    def is_unlikely_mother(self):
        """Uses a simple threshold to check whether it is likely that this mother is a mother cell."""
        return self.total() < 2

    def __str__(self):
        return str(self.total()) + " (based on " + str(self.__dict__["scores"]) + ")"

    def __repr__(self):
        return "Score(**" + repr(self.__dict__["scores"]) + ")"


class Family:
    """A mother cell with two daughter cells."""
    mother: Particle
    daughters: Set[Particle]  # Size of two, ensured by constructor.

    def __init__(self, mother: Particle, daughter1: Particle, daughter2: Particle):
        """Creates a new family. daughter1 and daughter2 can be swapped without consequences."""
        self.mother = mother
        self.daughters = {daughter1, daughter2}

    @staticmethod
    def _pos_str(particle: Particle) -> str:
        return "(" + ("%.2f" % particle.x) + ", " + ("%.2f" % particle.y) + ", " + ("%.0f" % particle.z) + ")"

    def __str__(self):
        return self._pos_str(self.mother) + " " + str(self.mother.time_point_number()) + "---> " \
               + " and ".join([self._pos_str(daughter) for daughter in self.daughters])

    def __repr__(self):
        return "Family(" + repr(self.mother) + ", " +  ", ".join([repr(daughter) for daughter in self.daughters]) + ")"

    def __hash__(self):
        hash_code = hash(self.mother)
        for daughter in self.daughters:
            hash_code += hash(daughter)
        return hash_code

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and other.mother == self.mother \
            and other.daughters == self.daughters


class ScoredFamily:
    """A family with a score attached. The higher the score, the higher the chance that this family is a "real" family,
    and not just some artifact."""
    family: Family
    score: Score

    def __init__(self, family: Family, score: Score):
        self.family = family
        self.score = score

    def __repr__(self):
        return "<" + str(self.family) + " scored " + str(self.score) + ">"


class TimePoint:
    """A single point in time. Particle positions & shapes, as well as possible families are stored here."""

    _time_point_number: int
    _particles: Dict[Particle, ParticleShape]
    _mother_scores: Dict[Family, Score]

    def __init__(self, time_point_number: int):
        self._time_point_number = time_point_number
        self._particles = dict()
        self._mother_scores = dict()

    def time_point_number(self) -> int:
        return self._time_point_number

    def particles(self) -> AbstractSet[Particle]:
        return self._particles.keys()

    def particles_and_shapes(self) -> Dict[Particle, ParticleShape]:
        return self._particles

    def mother_score(self, family: Family, score: Optional[Score] = None) -> Score:
        """Gets or sets the mother score of the given particle. Raises KeyError if no score has been set for this
         particle. Raises ValueError if you're looking in the wrong time point.
         """
        if family.mother.time_point_number() != self._time_point_number:
            raise ValueError("Family belongs to another time point")
        if score is not None:
            self._mother_scores[family] = score
            return score
        return self._mother_scores[family]

    def add_particle(self, particle: Particle) -> None:
        """Adds a particle to this time point. Does nothing if the particle was already added. Throws ValueError if
        the particle belongs to another time point. If the particle belongs to no time point, it is attached to this
        time point."""
        particle.with_time_point_number(self._time_point_number)
        if particle not in self._particles:
            self._particles[particle] = UnknownShape()

    def get_shape(self, particle: Particle) -> ParticleShape:
        """Gets the shape of a particle. Throws KeyError if the given particle is not part of this time point."""
        return self._particles[particle]

    def add_shaped_particle(self, particle: Particle, particle_shape: ParticleShape):
        """Adds a particle to this time point. If the particle was already added, its shape is replaced. Throws
        ValueError if the particle belongs to another time point. If the particle belongs to no time point, it is
        attached to this time point."""
        particle.with_time_point_number(self._time_point_number)
        self._particles[particle] = particle_shape

    def mother_scores(self, mother: Optional[Particle] = None) -> Iterable[ScoredFamily]:
        """Gets all mother scores of either all putative mothers, or just the given mother (if any)."""
        for family, score in self._mother_scores.items():
            if mother is not None:
                if family.mother != mother:
                    continue
            yield ScoredFamily(family, score)


class ImageLoader:
    """Responsible for loading all images in an experiment."""

    def load_3d_image(self, time_point: TimePoint) -> Optional[ndarray]:
        """Loads an image, usually from disk. Returns None if there is no image for this time point."""
        return None

    def unwrap(self) -> "ImageLoader":
        """If this loader is a wrapper around another loader, this method returns one loader below. Otherwise, it
        returns self.
        """
        return self


class _CachedImageLoader(ImageLoader):
    """Wrapper that caches the last few loaded images."""

    _internal: ImageLoader
    _image_cache: List

    def __init__(self, wrapped: ImageLoader):
        self._image_cache = []
        self._internal = wrapped

    def _add_to_cache(self, time_point_number: int, image: ndarray):
        if len(self._image_cache) > 5:
            self._image_cache.pop(0)
        self._image_cache.append((time_point_number, image))

    def load_3d_image(self, time_point: TimePoint) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        for entry in self._image_cache:
            if entry[0] == time_point_number:
                return entry[1]

        # Cache miss
        image = self._internal.load_3d_image(time_point)
        self._add_to_cache(time_point_number, image)
        return image

    def unwrap(self) -> ImageLoader:
        return self._internal


class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class ultimately collects all
    details of the experiment."""

    _time_points: Dict[str, TimePoint]
    _particle_links: Optional[Graph] = None
    _particle_links_baseline: Optional[Graph] = None # Links that are assumed to be correct
    _first_time_point_number: Optional[int] = None
    _last_time_point_number: Optional[int] = None
    _image_loader: ImageLoader = ImageLoader()

    def __init__(self):
        self._time_points = {}

    def add_particles_raw(self, time_point_number: int, raw_particles: List) -> None:
        """Adds particles to a time_point."""
        time_point = self.get_or_add_time_point(time_point_number)
        for raw_particle in raw_particles:
            particle = Particle(*raw_particle[0:3])
            particle_shape = shape.from_list(raw_particle[3:])
            time_point.add_shaped_particle(particle, particle_shape)

    def add_particle_raw(self, x: float, y: float, z: float, time_point_number: int):
        """Adds a single particle to the experiment, creating the time point if it does not exist yet."""
        time_point = self.get_or_add_time_point(time_point_number)
        time_point.add_particle(Particle(x, y, z))

    def add_particle(self, particle: Particle):
        """Adds a particle to the experiment. The particle must have a time point number specified."""
        time_point = self.get_or_add_time_point(particle.time_point_number())
        time_point.add_particle(particle)

    def get_time_point(self, time_point_number: int) -> TimePoint:
        """Gets the time point with the given number. Throws KeyError if no such time point exists."""
        return self._time_points[str(time_point_number)]

    def get_or_add_time_point(self, time_point_number: int) -> TimePoint:
        """Gets the time point with the given number. Creates the time point if it doesn't exist."""
        if time_point_number is None:
            raise ValueError("time_point_number is None")
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

    def time_points(self) -> Iterable[TimePoint]:
        first_number = self.first_time_point_number()
        last_number = self.last_time_point_number()
        current_number = first_number
        while current_number <= last_number:
            yield self.get_time_point(current_number)
            current_number += 1

    def set_image_loader(self, image_loader: ImageLoader):
        self._image_loader = _CachedImageLoader(image_loader)

    def get_image_loader(self):
        """Gets the image loader."""
        return self._image_loader.unwrap()  # The single unwrap call removes the cache

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        """Gets a stack of all images for a time point, one for every z layer. Returns None if there is no image."""
        return self._image_loader.load_3d_image(time_point)


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


def get_closest_n_particles(particles: Iterable[Particle], search_position: Particle, amount: int,
                            max_distance: int = 100000) -> Set[Particle]:
    max_distance_squared = max_distance ** 2
    closest_particles = []

    for particle in particles:
        distance_squared = particle.distance_squared(search_position)
        if distance_squared > max_distance_squared:
            continue
        if len(closest_particles) < amount or closest_particles[-1][0] > distance_squared:
            # Found closer particle
            closest_particles.append((distance_squared, particle))
            closest_particles.sort(key=itemgetter(0))
            if len(closest_particles) > amount:
                # List too long, remove furthest
                del closest_particles[-1]

    return_value = set()
    for distance_squared, particle in closest_particles:
        return_value.add(particle)
    return return_value