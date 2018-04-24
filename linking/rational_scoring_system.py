import math
from typing import Tuple, Optional

import cv2
import numpy
from numpy import ndarray

from core import Experiment, Particle, Score, TimePoint
from imaging import normalized_image, angles
from imaging.normalized_image import ImageEdgeError
from linking.link_fixer import get_2d_image
from linking.scoring_system import MotherScoringSystem


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
        daughter1_image_stack = experiment.get_image_stack(daughter_time_point)
        mother_image = mother_image_stack[int(mother.z)]
        mother_image_next = daughter1_image_stack[int(mother.z)]
        daughter1_image = daughter1_image_stack[int(daughter1.z)]
        daughter2_image = get_2d_image(experiment, daughter2)
        daughter1_image_prev = mother_image_stack[int(daughter1.z)]
        daughter2_image_prev = mother_image_stack[int(daughter2.z)]

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
            score_using_mother_shape(score, mother_time_point, mother, daughter1, daughter2)
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


def score_using_mother_shape(score: Score, time_point: TimePoint, mother: Particle, daughter1: Particle,
                             daughter2: Particle):
    """Returns a black-and-white image where white is particle and black is background, at least in theory."""
    score.mother_shape = 0
    score.mother_eccentric = 0
    score.daughters_side = 0

    # Zoom in on mother
    mother_shape = time_point.get_shape(mother)
    if mother_shape.is_unknown():
        return  # Too close to edge
    if not mother_shape.is_point_in_shape(0, 0):
        score.mother_eccentric = 2

    # Calculate the isoperimetric quotient and ellipse of the largest area
    area = mother_shape.area()
    perimeter = mother_shape.perimeter()
    isoperimetric_quotient = 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0
    if isoperimetric_quotient < 0.4:
        # Clear case of being a mother, give a bonus
        score.mother_shape = 2
    else:
        # Just use a normal scoring system
        score.mother_shape = 1 - isoperimetric_quotient

    # Score ellipsis
    if mother_shape.is_point_in_shape(0, 0):
        # Only try to score ellipsis if the mother position is inside the threshold
        # (Otherwise we got a very irregular shape of which the angle is unreliable)
        director = mother_shape.director(require_reliable=True)
        if director is not None:
            score.daughters_side = _score_daughter_sides(director, mother, daughter1, daughter2)


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
    score.daughters_area = 0

    daughter1_shape = time_point.get_shape(daughter1)
    daughter2_shape = time_point.get_shape(daughter2)

    if daughter1_shape.is_unknown() or daughter2_shape.is_unknown():
        return  # Too close to edge

    # Find contours with largest areas
    area1 = daughter1_shape.area()
    area2 = daughter2_shape.area()
    if min(area1, area2) / max(area1, area2) < 1/2:
        score.daughters_area = -1  # Size too dissimilar

    # Find daughters that are mirror images of themselves (mirror = equidistant line between the two)
    daughter1_direction = daughter1_shape.director()
    daughter2_direction = daughter1_shape.director()
    if angles.difference(daughter1_direction, daughter2_direction) > 90:
        daughter2_direction = angles.flipped(daughter2_direction)  # Make sure daughters lie in almost the same dir

    d1_to_d2_direction = angles.direction_2d(daughter1, daughter2)
    mirror_angle = angles.perpendicular(d1_to_d2_direction)
    mirrored_daughter2_direction = angles.mirrored(daughter2_direction, mirror_angle)
    difference = angles.difference(daughter1_direction, mirrored_daughter2_direction)
    if difference > 90:
        difference = 180 - difference  # Ignore whether daughter lies like this:  -->  or this:  <--
    score.daughters_angles = -difference / 90


def __get_threshold_for_shape(particle: Particle, full_image: ndarray, detection_radius: int) -> Optional[ndarray]:
    """Gets an image consisting of a circle or radius detection_radius around the given particle. Inside the circle,
    the image is thresholded using Otsu's method. This thresholded image can be used for shape detection.
    full_image must be a 2d ndarray consisting of intensities."""
    x = int(particle.x)
    y = int(particle.y)
    if x - detection_radius < 0 or y - detection_radius < 0 or x + detection_radius >= full_image.shape[1] \
            or y + detection_radius >= full_image.shape[0]:
        return None  # Out of bounds
    image = full_image[y - detection_radius:y + detection_radius, x - detection_radius:x + detection_radius]
    image_max_intensity = max(image.max(), 256)
    image_8bit = cv2.convertScaleAbs(image, alpha=256 / image_max_intensity, beta=0)
    __crop_to_circle(image_8bit)

    ret, thresholded_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image


def __crop_to_circle(image_8bit: ndarray):
    """Crops the (rectangular or square) 2D image to a circle. Essentially this function makes all pixels zero that are
    near the corners of the image.
    """
    image_circle = numpy.zeros_like(image_8bit)
    width = image_circle.shape[1]
    height = image_circle.shape[0]
    circle_size = min(width, height) - 3
    cv2.circle(image_circle, (int(width / 2), int(height / 2)), int(circle_size / 2), 255, -1)
    cv2.bitwise_and(image_8bit, image_circle, image_8bit)


def __find_largest_area(contours) -> Tuple[int, float]:
    highest_area = 0
    index_with_highest_area = -1
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > highest_area:
            highest_area = area
            index_with_highest_area = i
    return index_with_highest_area, highest_area