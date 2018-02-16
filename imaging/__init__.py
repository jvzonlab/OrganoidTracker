"""Images and positions of particles (biological cells in our case)"""
from typing import List, Iterable, Optional, Dict
from networkx import Graph
from imaging import image_cache


class Particle:

    x: float
    y: float
    z: float
    _frame_number: Optional[int]

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self._frame_number = None

    def distance_squared(self, other) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * 5) ** 2

    def frame_number(self):
        return self._frame_number

    def with_frame_number(self, frame_number: int):
        if self._frame_number is not None:
            raise ValueError("frame_number was already set")
        self._frame_number = int(frame_number)
        return self

    def __repr__(self):
        string = "Particle(" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.0f" % self.z) + ")"
        if self._frame_number is not None:
            string += ".with_frame_number(" + str(self._frame_number) + ")"
        return string

    def __str__(self):
        string = "cell at (" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.0f" % self.z) + ")"
        if self._frame_number is not None:
            string += " at time point " + str(self._frame_number)
        return string

    def __hash__(self):
        return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z)) ^ hash(int(self._frame_number))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and abs(self.x - other.x) < 0.00001 and abs(self.x - other.x) < 0.00001 and abs(self.z - other.z) < 0.00001 \
               and abs(self._frame_number - other._frame_number) < 0.00001


class Frame:
    """A single point in time."""

    _frame_number: int
    _particles: List[Particle]

    def __init__(self, frame_number: int):
        self._frame_number = frame_number
        self._particles = []
        self._image_loader = None

    def frame_number(self) -> int:
        return self._frame_number

    def particles(self) -> List[Particle]:
        return self._particles

    def add_particles(self, particles: Iterable[Particle]) -> None:
        """Adds all particles in the list to this frame. Throws ValueError if the particles were already assigned to
        a frame."""
        for particle in particles:
            particle.with_frame_number(self._frame_number)
            self._particles.append(particle)

    def set_image_loader(self, loader):
        """Sets the image loader. The image loader must ba a function with no args, that returns a numpy
        multidimensional array. Each element in the array is another array that forms an image.
        """
        self._image_loader = loader

    def load_images(self, allow_cache=True):
        if allow_cache:
            images = image_cache.get_from_cache(self._frame_number)
            if images is not None:
                return images

        # Cache miss
        images = self._load_images_uncached()
        if allow_cache:
            image_cache.add_to_cache(self._frame_number, images)
        return images

    def _load_images_uncached(self):
        image_loader = self._image_loader
        if self._image_loader is None:
            return None
        return image_loader()



class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class records the images, particle
     positions and particle trajectories."""

    _frames: Dict[str, Frame]
    _particle_links: Optional[Graph]
    _particle_links_baseline: Optional[Graph] # Links that are assumed to be correct
    _first_frame_number: Optional[int]
    _last_frame_number: Optional[int]

    def __init__(self):
        self._frames = {}
        self._particle_links = None
        self._particle_links_baseline = None
        self._last_frame_number = None
        self._first_frame_number = None

    def add_particles(self, frame_number: int, raw_particles) -> None:
        """Adds particles to a frame."""
        particles = []
        for raw_particle in raw_particles:
            particles.append(Particle(raw_particle[0], raw_particle[1], raw_particle[2]))
        frame = self._get_or_add_frame(frame_number)
        frame.add_particles(particles)

    def add_image_loader(self, frame_number: int, image_loader) -> None:
        frame = self._get_or_add_frame(frame_number)
        frame.set_image_loader(image_loader)

    def get_frame(self, frame_number: int) -> Frame:
        """Gets the frame with the given number. Throws KeyError if no such frame exists."""
        return self._frames[str(frame_number)]

    def _get_or_add_frame(self, frame_number: int) -> Frame:
        try:
            return self._frames[str(frame_number)]
        except KeyError:
            frame = Frame(frame_number)
            self._frames[str(frame_number)] = frame
            self._update_frame_statistics(frame_number)
            return frame

    def _update_frame_statistics(self, new_frame_number: int):
        if self._first_frame_number is None or self._first_frame_number > new_frame_number:
            self._first_frame_number = new_frame_number
        if self._last_frame_number is None or self._last_frame_number < new_frame_number:
            self._last_frame_number = new_frame_number

    def first_frame_number(self):
        if self._first_frame_number is None:
            raise ValueError("No frames exist")
        return self._first_frame_number

    def last_frame_number(self):
        if self._last_frame_number is None:
            raise ValueError("No frames exist")
        return self._last_frame_number

    def get_previous_frame(self, frame: Frame) -> Frame:
        """Gets the frame directly before the given frame, or KeyError if the given frame is the first frame."""
        return self.get_frame(frame.frame_number() - 1)

    def get_next_frame(self, frame: Frame) -> Frame:
        """Gets the frame directly after the given frame, or KeyError if the given frame is the last frame."""
        return self.get_frame(frame.frame_number() + 1)

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