from core import Experiment, Particle, Score


class MotherScoringSystem:
    """Abstract class for scoring putative mothers. See RationalScoringSystem for the default implementation. You can
    of course copy that class and make your own modifications. Don't forget to change the autotrack_create_links.py
    script and the autotrack_extract_mother_scores.py script to use your own class instead."""

    def calculate(self, experiment: Experiment, mother: Particle,
                  daughter1: Particle, daughter2: Particle) -> Score:
        """Gets the likeliness of this actually being a family. Higher is more likely."""
        pass
