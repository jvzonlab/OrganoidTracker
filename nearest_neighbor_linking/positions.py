"""Classes for expressing the positions of particles"""
import json


class Particle:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def link_to_previous(self, particle_in_past) -> None:
        """Makes a connection in time between the particles. Multiple connections are allowed. No checks for physical
        realism are done."""
        particle_in_past.links_next.append(self)
        self.links_previous.append(particle_in_past)

    def distance_squared(self, other) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * 10) ** 2;

    def __repr__(self):
        return "Particle(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"


class Frame:
    """A single point in time."""

    def __init__(self, frame_number: int, particles):
        self._frame_number = frame_number
        self._particles = particles

    def frame_number(self) -> int:
        return self._frame_number

    def particles(self):
        return self._particles


class Experiment:
    """A complete experiment, with many stacks of images collected over time"""
    def __init__(self):
        self._frames = {}

    def add_frame(self, frame_number: int, raw_particles) -> None:
        particles = []
        for raw_particle in raw_particles:
            particles.append(Particle(raw_particle[0], raw_particle[1], raw_particle[2]))
        self._frames[str(frame_number)] = Frame(frame_number, particles)

    def get_frame(self, frame_number : int) -> Frame:
        """Gets the frame with the given number. Throws ValueError if no such frame exists."""
        try:
            return self._frames[str(frame_number)]
        except KeyError:
            raise ValueError # More appropriate error

    def get_next_frame(self, frame: Frame) -> Frame:
        """Gets the frame directory after the given frame, or ValueError if the given frame is the last frame."""
        return self.get_frame(frame.frame_number() + 1)


def load_positions_from_json(json_file_name: str) -> Experiment:
    """Loads all positions from a JSON file"""
    experiment = Experiment()

    with open(json_file_name) as handle:
        frames = json.load(handle)
        for frame, raw_particles in frames.items():
            experiment.add_frame(int(frame), raw_particles)

    return experiment

