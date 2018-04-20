from core import Experiment, Particle, Score


class MotherScoringSystem:
    """Abstract class for scoring putative mothers."""

    def calculate(self, experiment: Experiment, mother: Particle,
                  daughter1: Particle, daughter2: Particle) -> Score:
        """Gets the likeliness of this actually being a family. Higher is more likely."""
        pass
