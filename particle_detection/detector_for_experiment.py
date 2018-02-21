from imaging import Experiment, Particle
from particle_detection import distance_transform_detection
from numpy import ndarray
from typing import Dict, Optional, List, Tuple


def detect_particles_using_distance_transform(experiment: Experiment, min_intensity = 0.6,
                                              max_cell_height = 6, **kwargs):
    for frame_number in range(experiment.first_frame_number(), experiment.last_frame_number() + 1):
        print("Searching for particles in frame " + str(frame_number) + "/"
              + str(experiment.last_frame_number()))
        frame = experiment.get_frame(frame_number)
        peaks_in_frame = {}
        image_stack = frame.load_images(allow_cache=False)
        for z in range(image_stack.shape[0]):
            image = image_stack[z]
            results = distance_transform_detection.perform(image, min_intensity=min_intensity)
            peaks_in_frame[z] = _to_peaks(results, z)

        particles = _to_particles(z_stop=image_stack.shape[0], peaks=peaks_in_frame, max_cell_height=max_cell_height)
        frame.add_particles(particles)


class Peak:
    x: int
    y: int
    z: int
    above: Optional["Peak"] = None
    converted: bool = False

    def __init__(self, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def to_particle(self, max_cell_height = 6) -> Optional[Particle]:
        """If this peak is the lowest in a stack, a particle is returned representing all those peaks."""
        if self.converted:
            return None
        count = 0
        total_x = 0
        total_y = 0
        total_z = 0
        peak = self
        i = 0
        while peak is not None and i <= max_cell_height:
            count += 1
            total_x += peak.x
            total_y += peak.y
            total_z += peak.z
            peak.converted = True

            peak = peak.above
            i += 1
        return Particle(total_x / count, total_y / count, total_z / count)


def _to_particles(z_stop: int, peaks: Dict[int, List[Peak]], max_cell_height) -> List[Particle]:
    # A particle is represented by one or more peaks vertically (z-direction) placed above each other

    # Find and connect peaks vertically, searching from bottom to top
    for z in range(z_stop - 1):
        _connect_peaks_vertically(peaks[z], peaks[z + 1])

    # Convert all peaks to particles
    particles = []
    for z in range(z_stop - 1):
        for peak in peaks[z]:
            particle = peak.to_particle(max_cell_height)
            if particle is not None:
                particles.append(particle)

    return particles


def _connect_peaks_vertically(peaks: List[Peak], peaks_above: List[Peak]):
    for peak in peaks:
        peak_above = _find_peak_near(peak, peaks_above)
        if peak_above is not None:
            peak.above = peak_above

def _find_peak_near(origin: Peak, search_in: List[Peak]) -> Optional[Peak]:
    for peak in search_in:
        if abs(peak.x - origin.x) <= 6 and abs(peak.y - origin.y) <= 6:
            return peak
    return None

def _to_peaks(results: ndarray, z: int) -> List[Peak]:
    return [Peak(x, y, z) for y, x in results]
