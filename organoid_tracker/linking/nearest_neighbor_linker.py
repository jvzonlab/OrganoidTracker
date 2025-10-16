"""Ultra-simple linker. Used as a starting point for more complex links."""
import math
from collections import defaultdict
from typing import Union, Tuple

import numpy
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.linking.nearby_position_finder import find_close_positions


class LogisticFit:
    """Result of the fit of the sigmoid function to the link-is-correct probability per distance data."""

    x0: float  # Midpoint, the distance at which the probability is 0.5
    k: float  # Steepness of the curve

    def __init__(self, popt: tuple[float, float]):
        self.x0 = popt[0]
        self.k = popt[1]

    def __call__(self, x: Union[float, numpy.ndarray]) -> float:
        return 1 - 1 / (1 + numpy.exp(-self.k * (x - self.x0)))


def nearest_neighbor(experiment: Experiment, *, tolerance: float = 1.0, back: bool = True, forward: bool = True,
                     max_distance_um: float = float("inf")) -> Links:
    """Simple nearest neighbour linking, keeping a list of potential candidates based on a given tolerance.

    A tolerance of 1.05 also links positions 5% from the closest position, so you end up with more links than you have
    positions.

    If the experiment has images loaded, then no links outside the images will be created.

    Nearest neighbor-linking can happen both forwards (every position is linked to the nearest in the next time point)
    and backwards (every position is linked to the nearest in the previous time point). If you do both, note that the
    tolerance is calculated independently for both directions: with a tolerance of for example 2, you'll get all forward
    links that are at most twice as long as the shortest forward link, and you'll get all backward links that are at
    most twice as long as the shortest backward link.
    """
    if not back and not forward:
        raise ValueError("Cannot create links if back and forward are both False.")
    links = Links()

    time_point_previous = None
    for time_point_current in experiment.time_points():

        if time_point_previous is not None:
            if back:
                _add_nearest_edges(links, experiment.positions, experiment.images, time_point_previous,
                                   time_point_current, tolerance=tolerance, max_distance_um=max_distance_um)
            if forward:
                _add_nearest_edges_extra(links, experiment.positions, experiment.images,
                                         time_point_previous, time_point_current, tolerance=tolerance,
                                         max_distance_um=max_distance_um)

        if time_point_current.time_point_number() % 50 == 0:
            print("    completed up to time point", time_point_current.time_point_number())
        time_point_previous = time_point_current

    print("Done creating nearest-neighbor links!")
    return links


def nearest_neighbor_probabilities(experiment: Experiment, *, tolerance: float = 2.0, max_distance_um: float = 50,
                                   distance_bin_size_um_log: float = 0.25) -> Tuple[Links, LogisticFit]:
    """First does nearest-neighbor linking, and assumes all nearest links to be true, the rest false. Creates a
    probability map of distance --> assumed correct. Then sets probabilities based on that.

    Default tolerance is 2.0, which means that all links that are at most twice as long as the nearest link are also
    taken into account as potential links.
    """
    links_nearest_neighbor = nearest_neighbor(experiment, tolerance=tolerance, back=True, forward=True,
                                              max_distance_um=max_distance_um)
    true_links_count_by_distance = defaultdict(lambda: 0)
    false_links_count_by_distance = defaultdict(lambda: 0)

    resolution = experiment.images.resolution()
    for time_point in experiment.positions.time_points():
        for position in experiment.positions.of_time_point(time_point):
            futures = links_nearest_neighbor.find_futures(position)
            if len(futures) == 0:
                continue  # No links, skip
            log_distances = numpy.log([position.distance_um(future, resolution) + 0.001 for future in futures])
            log_distances.sort()

            # First link is assumed true, the rest false
            true_links_count_by_distance[int(log_distances[0] // distance_bin_size_um_log)] += 1
            for log_distance in log_distances[1:]:
                distance_bin = int(log_distance // distance_bin_size_um_log)
                false_links_count_by_distance[distance_bin] += 1

    if len(true_links_count_by_distance) == 0 or len(false_links_count_by_distance) == 0:
        raise RuntimeError("No links were created, cannot create probability map.")

    # Now create a probability map based on the counts
    max_found_distance_key = max(max(true_links_count_by_distance.keys()), max(false_links_count_by_distance.keys()))
    probabilties_by_distance = {}
    for i in range(max_found_distance_key + 1):
        total = true_links_count_by_distance.get(i, 0) + false_links_count_by_distance.get(i, 0)
        if total == 0:
            probabilties_by_distance[i] = 0.0
        else:
            probabilties_by_distance[i] = true_links_count_by_distance.get(i, 0) / total

    # Add some extra bins with zero probability to help the fit
    # At large distances cells should never link, and at tiny distances they should always link.
    # But don't use exactly 0 or 1, as that doesn't work with the sigmoid fit.
    max_index = max(probabilties_by_distance.keys())
    min_index = min(probabilties_by_distance.keys())
    for i in range(max_index + 1, max_index + 10):
        probabilties_by_distance[i] = 0.001
    for i in range(min_index - 10, min_index - 1):
        probabilties_by_distance[i] = 0.999

    # Now set the probabilities in the links
    xdata = numpy.exp(numpy.array(list(probabilties_by_distance.keys()), dtype=float) * distance_bin_size_um_log)
    ydata = numpy.array(list(probabilties_by_distance.values()), dtype=float)

    def sigmoid(x, x0, k):
        y = 1 - 1 / (1 + numpy.exp(-k * (x - x0)))
        return y
    p0 = [numpy.median(xdata), 1]  # This is a mandatory initial guess
    popt, pcov = scipy.optimize.curve_fit(sigmoid, xdata, ydata, p0, method='lm')
    fit_sigmoid = LogisticFit(popt)

    for position_a, position_b in links_nearest_neighbor.find_all_links():
        distance_um = position_a.distance_um(position_b, resolution)
        probability = fit_sigmoid(distance_um)
        likelihood = math.log10(probability) - math.log10(1 - probability)
        links_nearest_neighbor.set_link_data(position_a, position_b, data_name="link_probability", value=probability)
        links_nearest_neighbor.set_link_data(position_a, position_b, data_name="link_penalty", value=-likelihood)
    return links_nearest_neighbor, fit_sigmoid


def _add_nearest_edges(links: Links, positions: PositionCollection, images: Images, time_point_previous: TimePoint,
                       time_point_current: TimePoint, *, tolerance: float, max_distance_um: float):
    """Adds edges pointing towards previous time point, making the shortest one the preferred."""
    resolution = images.resolution()
    for position in positions.of_time_point(time_point_current):
        # Check if position was inside the image in the previous time point
        previous_position = position.with_time_point(time_point_previous)
        if images.is_inside_image(previous_position) is False:
            # ^ Using "is False" because the method can also return None
            continue  # Skip, position will go out of view

        # If yes, make links to previous time point
        nearby_list = find_close_positions(positions.of_time_point(time_point_previous), around=position, max_amount=5,
                                           tolerance=tolerance, max_distance_um=max_distance_um, resolution=resolution)
        for nearby_position in nearby_list:
            links.add_link(position, nearby_position)


def _add_nearest_edges_extra(links: Links, positions: PositionCollection, images: Images, time_point_current: TimePoint,
                             time_point_next: TimePoint, *, tolerance: float, max_distance_um: float):
    """Adds edges to the next time point, which is useful if _add_edges missed some possible links."""
    resolution = images.resolution()
    for position in positions.of_time_point(time_point_current):
        # Check if position is still inside the image in the next time point
        next_position = position.with_time_point(time_point_next)
        if images.is_inside_image(next_position) is False:
            # ^ Using "is False" because the method can also return None
            continue  # Skip, position will go out of view

        # If yes, make links to next time point
        nearby_list = find_close_positions(positions.of_time_point(time_point_next), around=position, max_amount=5,
                                           tolerance=tolerance, max_distance_um=max_distance_um, resolution=resolution)
        for nearby_position in nearby_list:
            links.add_link(position, nearby_position)
