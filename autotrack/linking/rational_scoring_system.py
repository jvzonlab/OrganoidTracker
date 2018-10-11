import matplotlib.pyplot

import numpy
from numpy import ndarray

from autotrack.core.image_loader import ImageLoader
from autotrack.core.mask import create_mask_for, Mask, OutsideImageError
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.score import Score, Family
from autotrack.imaging import angles
from autotrack.linking.scoring_system import MotherScoringSystem


class RationalScoringSystem(MotherScoringSystem):
    """Rationally-designed score system."""

    def calculate(self, image_loader: ImageLoader, particle_shapes: ParticleCollection, family: Family) -> Score:
        mother = family.mother
        daughter1, daughter2 = family.daughters

        mother_image_stack = image_loader.get_image_stack(mother.time_point())
        daughter_image_stack = image_loader.get_image_stack(daughter1.time_point())

        mother_mask = _get_mask(mother_image_stack, mother, particle_shapes)
        daughter1_mask = _get_mask(daughter_image_stack, daughter1, particle_shapes)
        daughter2_mask = _get_mask(daughter_image_stack, daughter2, particle_shapes)

        try:
            mother_intensities = _get_nucleus_image(mother_image_stack, mother_mask)
            mother_intensities_next = _get_nucleus_image(daughter_image_stack, mother_mask)
            daughter1_intensities = _get_nucleus_image(daughter_image_stack, daughter1_mask)
            daughter2_intensities = _get_nucleus_image(daughter_image_stack, daughter2_mask)
            daughter1_intensities_prev = _get_nucleus_image(mother_image_stack, daughter1_mask)
            daughter2_intensities_prev = _get_nucleus_image(mother_image_stack, daughter2_mask)

            score = Score()
            score_mother_intensities(score, mother_intensities, mother_intensities_next)
            score_daughter_intensities(score, daughter1_intensities, daughter2_intensities,
                                       daughter1_intensities_prev, daughter2_intensities_prev)
            score_daughter_distances(score, mother, daughter1, daughter2)
            score_using_volumes(score, particle_shapes, mother, daughter1, daughter2)
            return score
        except OutsideImageError:
            print("No score for " + str(mother) + ": outside image")
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


def score_mother_intensities(score: Score, mother_intensities: ndarray, mother_intensities_next: ndarray):
    """Mother cell must have high intensity """

    # Intensity and contrast
    mean_value = numpy.nanmean(mother_intensities)
    variance_value = numpy.nanvar(mother_intensities)
    score.mother_contrast = variance_value * 10

    # Change of intensity (we use the max, as mothers often have both bright spots and darker spots near their center)
    mean_value_next = numpy.nanmean(mother_intensities_next)
    if mean_value / (mean_value_next + 0.0001) > 2:  # +0.0001 protects against division by zero
        score.mother_intensity_delta = 1
    elif mean_value / (mean_value_next + 0.0001) > 1.4:
        score.mother_intensity_delta = 0
    else:  # Intensity was almost constant over time (or even decreased), surely not a mother
        score.mother_intensity_delta = -1


def _score_daughter_sides(ellipse_angle: float, mother: Particle, daughter1: Particle,
                          daughter2: Particle) -> float:
    ellipse_angle_perpendicular = (ellipse_angle + 90) % 360

    # These angles are from 0 to 180, where 90 is completely aligned with the director of the ellipse
    daughter1_angle = angles.difference(angles.direction_2d(mother, daughter1), ellipse_angle_perpendicular)
    daughter2_angle = angles.difference(angles.direction_2d(mother, daughter2), ellipse_angle_perpendicular)

    if (daughter1_angle < 90 and daughter2_angle < 90) or (daughter1_angle > 90 and daughter2_angle > 90):
        return -1  # Two daughters on the same side, punish

    return 0


def score_using_volumes(score: Score, particles: ParticleCollection, mother: Particle, daughter1: Particle, daughter2: Particle):
    score.daughters_volume = 0

    mother_shape = particles.get_shape(mother)

    if mother_shape.is_unknown():
        score.mothers_volume = 2  # Unknown volume, so particle has an irregular shape. Likely a mother cell
    else:
        score.mothers_volume = 0

    daughter1_shape = particles.get_shape(daughter1)
    daughter2_shape = particles.get_shape(daughter2)

    if daughter1_shape.is_unknown() or daughter2_shape.is_unknown():
        return  # Too close to edge

    volume1 = daughter1_shape.volume()
    volume2 = daughter2_shape.volume()
    score.daughters_volume = min(volume1, volume2) / max(volume1, volume2)
    if score.daughters_volume < 0.75:
        score.daughters_volume = 0  # Almost surely not two daughter cells
    if mother_shape.volume() / (volume1 + volume2 + 0.0001) > 0.95:
        score.mothers_volume = 1

def _get_nucleus_image(image_stack: ndarray, mask: Mask) -> ndarray:
    """Gets the 2D image belonging to the particle. If the particle lays just above or below the image stack, the
    nearest image is returned."""
    return mask.create_masked_and_normalized_image(image_stack)


def _get_mask(image_stack: ndarray, particle: Particle, shapes: ParticleCollection) -> Mask:
    shape = shapes.get_shape(particle)
    mask = create_mask_for(image_stack)
    shape.draw_mask(mask, particle.x, particle.y, particle.z)
    return mask
