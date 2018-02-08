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
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * 10) ** 2;

    def frame_number(self, frame_number = None):
        if frame_number is not None:
            if self._frame_number is not None:
                # Frame number cannot be changed once set
                raise ValueError
            self._frame_number = frame_number
        return self._frame_number

    def __repr__(self):
        return "Particle(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def __str__(self):
        # Mostly useful for graphs
        return str(self._frame_number)


class Frame:
    """A single point in time."""

    def __init__(self, frame_number: int):
        self._frame_number = frame_number
        self._particles = []
        self._image_file_name = None

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

    def image_file_name(self, file: str = None) -> str:
        if file is not None:
            self._image_file_name = file
        return self._image_file_name


class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class records the images and particle
     positions."""

    _frames: List[Frame]
    _particle_links: Optional[Graph]

    def __init__(self):
        self._frames = {}
        self._particle_links = None

    def add_particles(self, frame_number: int, raw_particles) -> None:
        """Adds particles to a frame."""
        particles = []
        for raw_particle in raw_particles:
            particles.append(Particle(raw_particle[0], raw_particle[1], raw_particle[2]))
        frame = self._get_or_add_frame(frame_number)
        frame.add_particles(particles)

    def add_image(self, frame_number: int, image_file_name: str) -> None:
        frame = self._get_or_add_frame(frame_number)
        frame.image_file_name(image_file_name)

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
