from typing import Tuple, Dict

import cv2
import numpy
import math
from numpy import ndarray

from imaging import Experiment, Particle, normalized_image, angles
from linking.link_fixer import get_2d_image


class Score:
    """Represents a score, calculated from the individual elements. Usage:

        score = Score()
        score.foo = 4
        score.bar = 3
        # Results in score.total() == 7
    """

    def __init__(self, **kwargs):
        self.__dict__["scores"] = kwargs.copy()

    def __setattr__(self, key, value):
        self.__dict__["scores"][key] = value

    def __getattr__(self, item):
        return self.__dict__["scores"][item]

    def __delattr__(self, item):
        del self.__dict__["scores"][item]

    def total(self):
        score = 0
        for name, value in self.__dict__["scores"].items():
            score += value
        return score

    def __str__(self):
        return str(self.__scores)

    def __repr__(self):
        return "Score(**" + repr(self.__scores) + ")"


class MotherScoreSystem:
    """Abstract class for scoring putative mothers."""

    def calculate(self, experiment: Experiment, mother: Particle,
                  daughter1: Particle, daughter2: Particle) -> Score:
        """Gets the likeliness of this actually being a family. Higher is more likely."""
        pass


class RationalScoreSystem(MotherScoreSystem):
    """Rationally-designed score system."""
    mitotic_radius: int  # Used to detect max/min intensity of a cell. Can be a few pixels, just to suppress noise
    shape_detection_radius: int  # Used to detect the shape. Must be as large as the expected cell radius.

    def __init__(self, mitotic_radius: int, shape_detection_radius: int):
        self.mitotic_radius = mitotic_radius
        self.shape_detection_radius = shape_detection_radius

    def calculate(self, experiment: Experiment, mother: Particle, daughter1: Particle,
                  daughter2: Particle) -> Score:

        mother_image_stack = experiment.get_time_point(mother.time_point_number()).load_images()
        daughter1_image_stack = experiment.get_time_point(daughter1.time_point_number()).load_images()
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
            score_using_mother_shape(score, mother, daughter1, daughter2, mother_image, self.shape_detection_radius)
            return score
        except IndexError:
            return Score()


def score_daughter_distances(score: Score, mother: Particle, daughter1: Particle, daughter2: Particle):
    m_d1_distance = mother.distance_squared(daughter1)
    m_d2_distance = mother.distance_squared(daughter2)
    shorter_distance = m_d1_distance if m_d1_distance < m_d2_distance else m_d2_distance
    longer_distance = m_d1_distance if m_d1_distance > m_d2_distance else m_d2_distance
    if shorter_distance * (6 ** 2) < longer_distance:
        score.daughter_distances = -2
    score.daughter_distances = 0


def score_daughter_intensities(score: Score, daughter1_intensities: ndarray, daughter2_intensities: ndarray,
                               daughter1_intensities_prev: ndarray, daughter2_intensities_prev: ndarray):
    """Daughter cells must have almost the same intensity"""
    daughter1_average = numpy.average(daughter1_intensities)
    daughter2_average = numpy.average(daughter2_intensities)
    daughter1_average_prev = numpy.average(daughter1_intensities_prev)
    daughter2_average_prev = numpy.average(daughter2_intensities_prev)

    # Daughter cells must have almost the same intensity
    score.daughter_intensity_difference = -abs(daughter1_average - daughter2_average) / 2
    score.daughter_intensity_delta = 0
    if daughter1_average / (daughter1_average_prev + 0.0001) > 2:
        score.daughter_intensity_delta += 1
    if daughter2_average / (daughter2_average_prev + 0.0001) > 2:
        score.daughter_intensity_delta += 1


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


def score_using_mother_shape(score: Score, mother: Particle, daughter1: Particle, daughter2: Particle,
                             full_image: ndarray, detection_radius: int):
    """Returns a black-and-white image where white is particle and black is background, at least in theory."""
    score.mother_shape = 0
    score.daughter_angles = 0

    # Zoom in on mother
    x = int(mother.x)
    y = int(mother.y)
    if x - detection_radius < 0 or y - detection_radius < 0 or x + detection_radius >= full_image.shape[1] \
            or y + detection_radius >= full_image.shape[0]:
        return  # Out of bounds
    image = full_image[y - detection_radius:y + detection_radius, x - detection_radius:x + detection_radius]
    image_8bit = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
    __crop_to_circle(image_8bit)

    # Find contour
    ret, thresholded_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_image, contours, hierarchy = cv2.findContours(thresholded_image, 1, 2)

    # Calculate the isoperimetric quotient and ellipse of the largest area
    index_with_highest_area, area = __find_largest_area(contours)
    contour = contours[index_with_highest_area]
    perimeter = cv2.arcLength(contour, True)
    isoperimetric_quotient = 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0

    if thresholded_image[detection_radius, detection_radius] == 255:
        # Only try to score if the mother position is inside the threshold
        # (Otherwise we got a very irregular shape of which the angle is unreliable)
        fitted_ellipse = cv2.fitEllipse(contour)
        ellipse_length = fitted_ellipse[1][1]
        ellipse_width = fitted_ellipse[1][0]
        if ellipse_length / ellipse_width >= 1.2:
            score.daughter_angles = _score_daughter_angles(fitted_ellipse[2], mother, daughter1, daughter2)

    if isoperimetric_quotient < 0.4:
        # Clear case of being a mother, give a bonus
        score.mother_shape = 2
    else:
        # Just use a normal scoring system
        score.mother_shape = 1 - isoperimetric_quotient

    return score


def _score_daughter_angles(ellipse_angle: float, mother: Particle, daughter1: Particle,
                           daughter2: Particle) -> float:
    ellipse_angle_perpendicular = (ellipse_angle + 90) % 360

    # These angles are from 0 to 180, where 90 is completely aligned with the director of the ellipse
    daughter1_angle = angles.difference(angles.direction_2d(mother, daughter1), ellipse_angle_perpendicular)
    daughter2_angle = angles.difference(angles.direction_2d(mother, daughter2), ellipse_angle_perpendicular)

    if (daughter1_angle < 90 and daughter2_angle < 90) or (daughter1_angle > 90 and daughter2_angle > 90):
        return -1  # Two daughters on the same side, punish

    return 0


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