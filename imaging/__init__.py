"""Images and positions of particles (biological cells in our case)"""
from typing import List, Iterable, Optional
from networkx import Graph


class Particle:

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self._frame_number = None

    def distance_squared(self, other) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * 5) ** 2;

    def frame_number(self, frame_number = None):
        if frame_number is not None:
            if self._frame_number is not None:
                # Frame number cannot be changed once set
                raise ValueError
            self._frame_number = frame_number
        return self._frame_number

    def __repr__(self):
        return "Particle(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def __hash__(self):
        return hash(self.x) ^ hash(self.y) ^ hash(self.z) ^ hash(self._frame_number)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.x == other.x and self.y == other.y and self.z == other.z \
               and self._frame_number == other._frame_number

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
            particle.frame_number(self._frame_number)
            self._particles.append(particle)

    def set_image_loader(self, loader):
        """Sets the image loader. The image loader must ba a function with no args, that returns a numpy
        multidimensional array. Each element in the array is another array that forms an image.
        """
        self._image_loader = loader

    def load_images(self):
        image_loader = self._image_loader
        if self._image_loader is None:
            return []
        return image_loader()


class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class records the images and particle
     positions."""

    _frames: List[Frame]
    _particle_links: Optional[Graph]
    _particle_links_baseline: Optional[Graph] # Links that are assumed to be correct

    def __init__(self):
        self._frames = {}
        self._particle_links = None
        self._particle_links_baseline = None

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
            return frame

    def get_next_frame(self, frame: Frame) -> Frame:
        """Gets the frame directory after the given frame, or KeyError if the given frame is the last frame."""
        return self.get_frame(frame.frame_number() + 1)

    def particle_links(self, network: Optional[Graph] = None) -> Optional[Graph]:
        """Gets or sets the particle linking results. It is not possible to replace exising results."""
        if network is not None:
            if self._particle_links is not None:
                raise ValueError # Cannot replace network
            self._particle_links = network
        return self._particle_links

    def particle_links_baseline(self, network: Optional[Graph] = None) -> Optional[Graph]:
        """Gets or sets a particle linking result **that is known to be correct**."""
        if network is not None:
            if self._particle_links_baseline is not None:
                raise ValueError # Cannot replace network
            self._particle_links_baseline = network
        return self._particle_links_baseline