"""Designed scoring system for scoring putative mother cells."""
import traceback

import numpy
from numpy import ndarray

from autotrack.core.images import Images, Image
from autotrack.core.mask import create_mask_for, Mask, OutsideImageError
from autotrack.core.position import Position
from autotrack.core.position_collection import PositionCollection
from autotrack.core.resolution import ImageResolution
from autotrack.core.score import Score, Family
from autotrack.linking.scoring_system import MotherScoringSystem


class RationalScoringSystem(MotherScoringSystem):
    """Rationally-designed score system."""

    def calculate(self, images: Images, position_shapes: PositionCollection, family: Family) -> Score:
        mother = family.mother
        daughter1, daughter2 = family.daughters

        mother_image_stack = images.get_image(mother.time_point())
        daughter_image_stack = images.get_image(daughter1.time_point())

        try:
            mother_mask = _get_mask(mother_image_stack, mother, position_shapes)
            daughter1_mask = _get_mask(daughter_image_stack, daughter1, position_shapes)
            daughter2_mask = _get_mask(daughter_image_stack, daughter2, position_shapes)

            mother_intensities = _get_nucleus_image(mother_image_stack, mother_mask)
            mother_intensities_next = _get_nucleus_image(daughter_image_stack, mother_mask)
            daughter1_intensities = _get_nucleus_image(daughter_image_stack, daughter1_mask)
            daughter2_intensities = _get_nucleus_image(daughter_image_stack, daughter2_mask)
            daughter1_intensities_prev = _get_nucleus_image(mother_image_stack, daughter1_mask)
            daughter2_intensities_prev = _get_nucleus_image(mother_image_stack, daughter2_mask)

            score = Score()
            score_mother_intensities(score, mother, mother_intensities, mother_intensities_next)
            score_daughter_intensities(score, daughter1_intensities, daughter2_intensities,
                                       daughter1_intensities_prev, daughter2_intensities_prev)
            score_daughter_distances(score, mother, daughter1, daughter2, images.resolution())
            score_using_volumes(score, position_shapes, mother, daughter1, daughter2)
            return score
        except OutsideImageError:
            print("No score for " + str(mother) + ": outside image")
            return Score()


def score_daughter_distances(score: Score, mother: Position, daughter1: Position, daughter2: Position,
                             resolution: ImageResolution):
    m_d1_distance = mother.distance_squared(daughter1, resolution)
    m_d2_distance = mother.distance_squared(daughter2, resolution)
    shorter_distance = m_d1_distance if m_d1_distance < m_d2_distance else m_d2_distance
    longer_distance = m_d1_distance if m_d1_distance > m_d2_distance else m_d2_distance
    if shorter_distance * (6 ** 2) < longer_distance:
        score.daughters_distance = -2
    else:
        score.daughters_distance = 0


def score_daughter_intensities(score: Score, daughter1_intensities: ndarray, daughter2_intensities: ndarray,
                               daughter1_intensities_prev: ndarray, daughter2_intensities_prev: ndarray):
    """Daughter cells must have almost the same intensity"""
    daughter1_average = numpy.nanmean(daughter1_intensities)
    daughter2_average = numpy.nanmean(daughter2_intensities)
    daughter1_average_prev = numpy.nanmean(daughter1_intensities_prev)
    daughter2_average_prev = numpy.nanmean(daughter2_intensities_prev)

    # Daughter cells must have almost the same intensity
    score.daughters_intensity_difference = -abs(daughter1_average - daughter2_average) / 2
    score.daughters_intensity_delta = 1
    if daughter1_average / (daughter1_average_prev + 0.0001) > 2:
        score.daughters_intensity_delta += 1
    elif daughter1_average / (daughter1_average_prev + 0.0001) < 1:
        score.daughters_intensity_delta -= 1
    if daughter2_average / (daughter2_average_prev + 0.0001) > 2:
        score.daughters_intensity_delta += 1
    elif daughter2_average / (daughter2_average_prev + 0.0001) < 1:
        score.daughters_intensity_delta -= 1


def score_mother_intensities(score: Score, mother: Position, mother_intensities: ndarray, mother_intensities_next: ndarray):
    """Mother cell must have high intensity """

    # Intensity and contrast
    mean_value = numpy.nanmean(mother_intensities)
    variance_value = numpy.nanvar(mother_intensities)
    if numpy.isnan(mean_value) or numpy.isnan(variance_value):
        score.mother_contrast = 0
        score.mother_intensity_delta = 0
        return
    score.mother_contrast = variance_value * 10

    # Change of intensity (we use the max, as mothers often have both bright spots and darker spots near their center)
    mean_value_next = numpy.nanmean(mother_intensities_next)
    if mean_value / (mean_value_next + 0.0001) > 2:  # +0.0001 protects against division by zero
        score.mother_intensity_delta = 1
    elif mean_value / (mean_value_next + 0.0001) > 1.4:
        score.mother_intensity_delta = 0
    else:  # Intensity was almost constant over time (or even decreased), surely not a mother
        score.mother_intensity_delta = -1


def score_using_volumes(score: Score, positions: PositionCollection, mother: Position, daughter1: Position, daughter2: Position):
    score.daughters_volume = 0

    mother_shape = positions.get_shape(mother)

    score.mother_volume = -10
    if mother_shape.is_unknown():
        return

    daughter1_shape = positions.get_shape(daughter1)
    daughter2_shape = positions.get_shape(daughter2)

    if daughter1_shape.is_unknown() or daughter2_shape.is_unknown():
        return  # Too close to edge

    volume1 = daughter1_shape.volume()
    volume2 = daughter2_shape.volume()
    score.daughters_volume = min(volume1, volume2) / max(volume1, volume2)
    if score.daughters_volume < 0.75:
        score.daughters_volume = 0  # Almost surely not two daughter cells
    if mother_shape.volume() / (volume1 + volume2 + 0.0001) > 0.95:
        score.mother_volume = 0  # We have a mother cell, or maybe just a big cell


def _get_nucleus_image(image_stack: Image, mask: Mask) -> ndarray:
    """Gets the 2D image belonging to the position. If the position lays just above or below the image stack, the
    nearest image is returned."""
    return mask.create_masked_and_normalized_image(image_stack)


def _get_mask(image_stack: Image, position: Position, shapes: PositionCollection) -> Mask:
    shape = shapes.get_shape(position)
    mask = create_mask_for(image_stack)
    shape.draw_mask(mask, position.x, position.y, position.z )
    return mask
