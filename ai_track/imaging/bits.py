import cv2
from numpy import ndarray


def image_to_8bit(image: ndarray):
    return cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
