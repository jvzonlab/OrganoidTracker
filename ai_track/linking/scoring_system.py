"""Base class for scoring system for putative mother cells."""

from ai_track.core.image_loader import ImageLoader
from ai_track.core.images import Images
from ai_track.core.position_collection import PositionCollection
from ai_track.core.position import Position
from ai_track.core.score import Score, Family


class MotherScoringSystem:
    """Abstract class for scoring putative mothers. See RationalScoringSystem for the default implementation. You can
    of course copy that class and make your own modifications. Don't forget to change the ai_track_create_links.py
    script and the ai_track_extract_mother_scores.py script to use your own class instead."""

    def calculate(self, images: Images, position_shapes: PositionCollection, family: Family) -> Score:
        """Gets the likeliness of this actually being a family. Higher is more likely."""
        raise NotImplementedError()
