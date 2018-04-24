import cv2
from typing import Optional, Tuple

import numpy

from core import Experiment, Particle
from numpy import ndarray

from core.shape import EllipseShape


def detect_for_all(experiment: Experiment, detection_radius: int):
    for time_point in experiment.time_points():
        print("Working on time point " + str(time_point.time_point_number()))
        image_stack = experiment.get_image_stack(time_point)
        for particle in time_point.particles():
            image = image_stack[int(particle.z)]
            thresholded = __get_threshold_for_shape(particle, image, detection_radius)
            if thresholded is None:
                continue  # Too close to edge of image

            contour_image, contours, hierarchy = cv2.findContours(thresholded, 1, 2)
            contour_index, area = __find_contour_with_largest_area(contours)
            if contour_index == -1 or area < 20:
                continue  # No contours found
            ellipse = cv2.fitEllipse(contours[contour_index])
            ellipse_shape = EllipseShape(ellipse_dx=ellipse[0][0] - detection_radius,
                                         ellipse_dy=ellipse[0][1] - detection_radius,
                                         ellipse_width=ellipse[1][0],
                                         ellipse_height=ellipse[1][1],
                                         ellipse_angle=ellipse[2],
                                         original_area=area,
                                         original_perimeter=cv2.arcLength(contours[contour_index], True))
            time_point.add_shaped_particle(particle, ellipse_shape)


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
    thresholded_image = cv2.erode(thresholded_image, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
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


def __find_contour_with_largest_area(contours) -> Tuple[int, float]:
    highest_area = 0
    index_with_highest_area = -1
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > highest_area:
            highest_area = area
            index_with_highest_area = i
    return index_with_highest_area, highest_area