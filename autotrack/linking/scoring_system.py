from autotrack.core.image_loader import ImageLoader
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.score import Score, Family


class MotherScoringSystem:
    """Abstract class for scoring putative mothers. See RationalScoringSystem for the default implementation. You can
    of course copy that class and make your own modifications. Don't forget to change the autotrack_create_links.py
    script and the autotrack_extract_mother_scores.py script to use your own class instead."""

    def calculate(self, image_loader: ImageLoader, particle_shapes: ParticleCollection, family: Family) -> Score:
        """Gets the likeliness of this actually being a family. Higher is more likely."""
        raise NotImplementedError()
