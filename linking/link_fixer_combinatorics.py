import numpy
from networkx import Graph
from numpy import ndarray

from imaging import Particle, Experiment, TimePoint, normalized_image, Marker
from linking.link_fixer import fix_no_future_particle, with_only_the_preferred_edges
from linking_analysis import logical_tests
from typing import Set, Union, Tuple, Dict


class LinkingResult(Marker):
    mothers: Dict[Particle, int]
    mothers_unsure: Dict[Particle, int]
    daughters: Dict[Particle, int]
    daughters_unsure: Dict[Particle, int]

    def __init__(self):
        self.mothers = dict()
        self.mothers_unsure = dict()
        self.daughters = dict()
        self.daughters_unsure = dict()

    def __str__(self):
        return "Mothers: " + str(len(self.mothers)) + "(+" + str(len(self.mothers_unsure)) +" unsure)" \
               + " Daughters: " + str(len(self.daughters)) + "(+" + str(len(self.daughters_unsure)) +"unsure)"

    def add_daughter(self, time_point: TimePoint, particle: Particle):
        self.daughters[particle] = time_point.time_point_number()

    def add_mother(self, time_point: TimePoint, particle: Particle):
        self.mothers[particle] = time_point.time_point_number()

    def add_mother_unsure(self, time_point: TimePoint, particle: Particle):
        self.mothers_unsure[particle] = time_point.time_point_number()

    def add_daughter_unsure(self, time_point: TimePoint, particle: Particle):
        self.daughters_unsure[particle] = time_point.time_point_number()

    def get_color(self, time_point: TimePoint, particle: Particle) -> Union[None, str, Tuple]:
        if particle in self.mothers and self.mothers[particle] == time_point.time_point_number():
            return "lime"
        if particle in self.mothers_unsure and self.mothers_unsure[particle] == time_point.time_point_number():
            return "xkcd:mint"
        if particle in self.daughters and self.daughters[particle] == time_point.time_point_number():
            return "red"
        if particle in self.daughters_unsure and self.daughters_unsure[particle] == time_point.time_point_number():
            return "xkcd:coral"
        return None

def prune_links(experiment: Experiment, graph: Graph, detection_radius_small: int, detection_radius_large: int) -> LinkingResult:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next time_point). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """

    linking_result = LinkingResult()

    [fix_no_future_particle(graph, particle) for particle in graph.nodes()]
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        time_point = experiment.get_time_point(time_point_number)
        _fix_cell_divisions(experiment, graph, time_point, linking_result, detection_radius_small, detection_radius_large)
    [fix_no_future_particle(graph, particle) for particle in graph.nodes()]

    graph = with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return linking_result


def _fix_cell_divisions(experiment: Experiment, graph: Graph, time_point: TimePoint, linking_result: LinkingResult,
                        detection_radius_small: int, detection_radius_large: int):
    print("Working on time point " + str(time_point.time_point_number()))
    tp_images = time_point.load_images()
    try:
        next_time_point = experiment.get_next_time_point(time_point)
    except KeyError:
        return set()  # Last time point, cannot do anything
    next_tp_images = next_time_point.load_images()

    for particle in time_point.particles():
        _search_for_mother(particle, time_point, tp_images, next_tp_images, linking_result,
                           detection_radius_small, detection_radius_large)

    for particle in next_time_point.particles():
        _search_for_daugher(particle, time_point, tp_images, next_tp_images, linking_result,
                           detection_radius_small, detection_radius_large)


def _search_for_mother(particle: Particle, time_point: TimePoint, tp_images: ndarray, next_tp_images: ndarray,
                      linking_result: LinkingResult, detection_radius_small: int, detection_radius_large: int):
   z = int(particle.z)
   try:
       intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                          detection_radius_small)
       intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                      detection_radius_small)
       if numpy.average(intensities_previous) / (numpy.average(intensities_next) + 0.0001) > 2:
           linking_result.add_mother(time_point, particle)
           return
       intensities_difference = intensities_next - intensities_previous
       if intensities_difference.max() < 0.05 and intensities_difference.min() > -0.05:
           return  # No change in intensity, so just a normal moving cell

       intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                          detection_radius_large)
       intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                      detection_radius_large)
       if numpy.average(intensities_previous) / (numpy.average(intensities_next) + 0.0001) > 1.1:
           linking_result.add_mother_unsure(time_point, particle)
   except IndexError:
       pass  # Cell too close to border of image


def _search_for_daugher(particle: Particle, time_point: TimePoint, tp_images: ndarray, next_tp_images: ndarray,
                   linking_result: LinkingResult, detection_radius_small: int, detection_radius_large: int):
    z = int(particle.z)
    try:
        intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                       detection_radius_small)
        intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                           detection_radius_small)
        intensities_difference = intensities_next - intensities_previous
        if numpy.average(intensities_next) / (numpy.average(intensities_previous) + 0.0001) > 2:
            linking_result.add_daughter(time_point, particle)
            return
        if intensities_difference.max() < 0.05 and intensities_difference.min() > -0.05:
            return  # No change in intensity, so just a normal moving cell

        intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                       detection_radius_large)
        intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                           detection_radius_large)
        intensities_difference = intensities_next - intensities_previous
        if intensities_difference.max() > 0.4:
            linking_result.add_daughter_unsure(time_point, particle)
    except IndexError:
        pass  # Cell too close to border of image