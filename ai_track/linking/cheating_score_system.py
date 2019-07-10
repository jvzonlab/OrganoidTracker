"""Artificial scoring system for testing purposes."""

from typing import Set, Iterable

from ai_track.core.images import Images
from ai_track.core.position_collection import PositionCollection
from ai_track.core.score import Family, Score
from ai_track.linking.scoring_system import MotherScoringSystem


class CheatingScoringSystem(MotherScoringSystem):
    """This scoring system cheats: it returns 10 for all known mothers (specified in advance) and 0 for all other cells.
    This scoring system is useful for testing how an ideal scoring system would behave."""

    YES_SCORE = Score(mother=10)
    NO_SCORE = Score(mother=0)

    _families: Set[Family]

    def __init__(self, families: Iterable[Family]):
        """Initializer. families is the ground-thruth of all families."""
        self._families = set(families)

    def calculate(self, images: Images, position_shapes: PositionCollection, family: Family) -> Score:
        if family in self._families:
            return CheatingScoringSystem.YES_SCORE
        else:
            return CheatingScoringSystem.NO_SCORE
