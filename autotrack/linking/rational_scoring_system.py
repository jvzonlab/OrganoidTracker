import math

import numpy
from numpy import ndarray

from autotrack.core import Experiment, Particle, Score, TimePoint
from autotrack.imaging import angles, normalized_image
from autotrack.imaging.normalized_image import ImageEdgeError
from autotrack.linking.link_fixer import get_2d_image_from_experiment
from autotrack.linking.scoring_system import MotherScoringSystem


class RationalScoringSystem(MotherScoringSystem):
    """Rationally-designed score system."""
    mitotic_radius: int  # Used to detect max/min intensity of a cell. Can be a few pixels, just to suppress noise

    def __init__(self, mitotic_radius: int):
        self.mitotic_radius = mitotic_radius

    def calculate(self, experiment: Experiment, mother: Particle, daughter1: Particle,
                  daughter2: Particle) -> Score:
        mother_time_point = experiment.get_time_point(mother.time_point_number())
        daughter_time_point = experiment.get_time_point(daughter1.time_point_number())
        mother_image_stack = experiment.get_image_stack(mother_time_point)
        daughter_image_stack = experiment.get_image_stack(daughter_time_point)

        mother_image = _get_2d_image(mother_image_stack, mother)
        mother_image_next = _get_2d_image(daughter_image_stack, mother)
        daughter1_image = _get_2d_image(daughter_image_stack, daughter1)
        daughter2_image = _get_2d_image(daughter_image_stack, daughter2)
        daughter1_image_prev = _get_2d_image(mother_image_stack, daughter1)
        daughter2_image_prev = _get_2d_image(mother_image_stack, daughter2)

        try:
            mother_intensities = normalized_image.get_square(mother_image, mother.x, mother.y, self.mitotic_radius)
            mother_intensities_next = normalized_image.get_square(mother_image_next, mother.x, mother.y,
                                                                  self.mitotic_radius)
            daughter1_intensities = normalized_image.get_square(daughter1_image, daughter1.x, daughter1.y,
                                                                self.mitotic_radius)
            daughter2_intensities = normalized_image.get_square(daughter2_image, daughter2.x, daughter2.y,
                                                                self.mitotic_radius)
            daughter1_intensities_prev = normalized_image.get_square(daughter1_image_prev, daughter1.x, daughter1.y,
                                                                     self.mitotic_radius)
            daughter2_intensities_prev = normalized_image.get_square(daughter2_image_prev, daughter2.x, daughter2.y,
                                                                     self.mitotic_radius)

            score = Score()
            score_mother_intensities(score, mother_intensities, mother_intensities_next)
            score_daughter_intensities(score, daughter1_intensities, daughter2_intensities,
                                                daughter1_intensities_prev, daughter2_intensities_prev)
            score_daughter_distances(score, mother, daughter1, daughter2)
            score_using_daughter_shapes(score, daughter_time_point, daughter1, daughter2)
            return score
        except ImageEdgeError:
            return Score()


def score_daughter_distances(score: Score, mother: Particle, daughter1: Particle, daughter2: Particle):
    m_d1_distance = mother.distance_squared(daughter1)
    m_d2_distance = mother.distance_squared(daughter2)
    shorter_distance = m_d1_distance if m_d1_distance < m_d2_distance else m_d2_distance
    longer_distance = m_d1_distance if m_d1_distance > m_d2_distance else m_d2_distance
    if shorter_distance * (6 ** 2) < longer_distance:
        score.daughters_distance = -2
    else:
        score.daughters_distance = 0


def score_daughter_intensities(score: Score, daughter1_intensities: ndarray, daughter2_intensities: ndarray,
                               daughter1_intensities_prev: ndarray, daughter2_intensities_prev: ndarray):
    """Daughter cells must have almost the same intensity"""
    daughter1_average = numpy.average(daughter1_intensities)
    daughter2_average = numpy.average(daughter2_intensities)
    daughter1_average_prev = numpy.average(daughter1_intensities_prev)
    daughter2_average_prev = numpy.average(daughter2_intensities_prev)

    # Daughter cells must have almost the same intensity
    score.daughters_intensity_difference = -abs(daughter1_average - daughter2_average) / 2
    score.daughters_intensity_delta = 0
    if daughter1_average / (daughter1_average_prev + 0.0001) > 2:
        score.daughters_intensity_delta += 1
    if daughter2_average / (daughter2_average_prev + 0.0001) > 2:
        score.daughters_intensity_delta += 1


def score_mother_intensities(score: Score, mother_intensities: ndarray, mother_intensities_next: ndarray):
    """Mother cell must have high intensity """

    # Intensity and contrast
    min_value = numpy.min(mother_intensities)
    max_value = numpy.max(mother_intensities)
    score.mother_intensity = 1 if max_value > 0.7 else 0  # The higher intensity, the better: the DNA is concentrated
    score.mother_contrast = 0.5 if max_value - min_value > 0.4 else 0  # High contrast is also desirable

    # Change of intensity (we use the max, as mothers often have both bright spots and darker spots near their center)
    max_value_next = numpy.max(mother_intensities_next)
    if max_value / (max_value_next + 0.0001) > 2:  # +0.0001 protects against division by zero
        score.mother_intensity_delta = 1
    else:
        score.mother_intensity_delta = 0


def _score_daughter_sides(ellipse_angle: float, mother: Particle, daughter1: Particle,
                          daughter2: Particle) -> float:
    ellipse_angle_perpendicular = (ellipse_angle + 90) % 360

    # These angles are from 0 to 180, where 90 is completely aligned with the director of the ellipse
    daughter1_angle = angles.difference(angles.direction_2d(mother, daughter1), ellipse_angle_perpendicular)
    daughter2_angle = angles.difference(angles.direction_2d(mother, daughter2), ellipse_angle_perpendicular)

    if (daughter1_angle < 90 and daughter2_angle < 90) or (daughter1_angle > 90 and daughter2_angle > 90):
        return -1  # Two daughters on the same side, punish

    return 0


def score_using_daughter_shapes(score: Score, time_point: TimePoint, daughter1: Particle, daughter2: Particle):
    score.daughters_angles = 0
    score.daughters_volume = 0

    daughter1_shape = time_point.get_shape(daughter1)
    daughter2_shape = time_point.get_shape(daughter2)

    if daughter1_shape.is_unknown() or daughter2_shape.is_unknown():
        return  # Too close to edge

    # Find contours with largest areas
    volume1 = daughter1_shape.volume()
    volume2 = daughter2_shape.volume()
    if min(volume1, volume2) / max(volume1, volume2) < 1/2:
        score.daughters_volume = -1  # Size too dissimilar


def _get_2d_image(image_stack: ndarray, particle: Particle) -> ndarray:
    """Gets the 2D image belonging to the particle. If the particle lays just above or below the image stack, the
    nearest image is returned."""
    z = int(particle.z)
    if z == len(image_stack):
        z = len(image_stack) - 1
    if z == -1:
        z = 0
    return image_stack[z]