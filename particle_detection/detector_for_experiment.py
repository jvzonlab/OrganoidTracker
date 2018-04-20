import math
from typing import Dict, Optional, List, Any, Iterable

from numpy import ndarray

from core import Experiment, Particle
from particle_detection import Detector


def detect_particles_in_3d(experiment: Experiment, method: Detector, max_cell_height=6, **kwargs):
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        print("Searching for particles in time_point " + str(time_point_number) + "/"
              + str(experiment.last_time_point_number()))
        time_point = experiment.get_time_point(time_point_number)
        peaks_in_time_point = {}
        image_stack = experiment.get_image_stack(time_point)
        for z in range(image_stack.shape[0]):
            image = image_stack[z]
            results = method.detect(image, **kwargs)
            peaks_in_time_point[z] = _to_peaks(results, z)

        particles = _to_particles(z_stop=image_stack.shape[0], peaks=peaks_in_time_point, max_cell_height=max_cell_height)
        for particle in particles:
            time_point.add_particle(particle)


class Peak:
    """A higher signal, usually indicative of a particle. Signals stacked above each other usually represent one
     particle, and this class accounts for that.
     """
    x: int
    y: int
    z: int
    above: Optional["Peak"] = None
    converted: bool = False

    def __init__(self, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def to_particles(self, max_cell_height = 6) -> List[Particle]:
        """If this peak is the lowest in a stack, a particle is returned representing all those peaks. If there are too
        many peaks for max_cell_height, the stack is split into roughtly equal parts with a size <= max_cell_height
        """
        if self.converted:
            return []

        peaks = self._grab_all_peaks_above()
        desired_chunk_amount = int(math.ceil(len(peaks) / max_cell_height))
        chunked_peaks = _chunk(peaks, desired_chunk_amount)
        return [_average(chunk) for chunk in chunked_peaks]

    def _grab_all_peaks_above(self):
        peaks = []
        peak = self
        while peak is not None:
            peaks.append(peak)
            peak.converted = True

            peak = peak.above
        return peaks


def _average(peaks: List[Peak]) -> Particle:
    """Averages the x,y,z of the peaks to obtain a single particle"""
    x_total = 0
    y_total = 0
    z_total = 0
    for peak in peaks:
        x_total += peak.x
        y_total += peak.y
        z_total += peak.z
    return Particle(x_total / len(peaks), y_total / len(peaks), z_total / len(peaks))


def _chunk(seq: List, num: int) -> List[List]:
    """Divides a list into num roughly equal parts. So `_chunk(range(11), 3)` becomes
    `[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]]`
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def _to_particles(z_stop: int, peaks: Dict[int, List[Peak]], max_cell_height) -> Iterable[Particle]:
    # A particle is represented by one or more peaks vertically (z-direction) placed above each other

    # Find and connect peaks vertically, searching from bottom to top
    for z in range(z_stop - 1):
        _connect_peaks_vertically(peaks[z], peaks[z + 1])

    # Convert all peaks to particles
    particles = []
    for z in range(z_stop - 1):
        for peak in peaks[z]:
            yield peak.to_particles(max_cell_height)


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


def get_file_name(method: Detector, method_parameters: Dict[str, Any]):
    """Gets a unique file name based on the given method and its parameters"""
    file_name = method.__class__.__name__.replace("Detector", "")
    for key, value in method_parameters.items():
        file_name += " " + key + "=" + str(value)
    return file_name + ".json"
