"""Artificial scoring system for testing purposes."""

from typing import Set, Iterable

from organoid_tracker.core.images import Images
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.score import Family, Score
from organoid_tracker.linking.scoring_system import MotherScoringSystem


class CheatingScoringSystem(MotherScoringSystem):
    """This scoring system cheats: it returns 10 for all known mothers (specified in advance) and 0 for all other cells.
    This scoring system is useful for testing how an ideal scoring system would behave."""

    YES_SCORE = Score(mother=10)
    NO_SCORE = Score(mother=0)

    _families: Set[Family]

    def __init__(self, families: Iterable[Family]):
        """Initializer. families is the ground-thruth of all families."""
        self._families = set(families)

    def calculate(self, images: Images, position_data: PositionData, family: Family) -> Score:
        if family in self._families:
            return CheatingScoringSystem.YES_SCORE
        else:
            return CheatingScoringSystem.NO_SCORE
