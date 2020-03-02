"""Base class for scoring system for putative mother cells."""

from organoid_tracker.core.image_loader import ImageLoader
from organoid_tracker.core.images import Images
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.score import Score, Family


class MotherScoringSystem:
    """Abstract class for scoring putative mothers. See RationalScoringSystem for the default implementation. You can
    of course copy that class and make your own modifications. Don't forget to change the organoid_tracker_create_links.py
    script and the organoid_tracker_extract_mother_scores.py script to use your own class instead."""

    def calculate(self, images: Images, position_shapes: PositionCollection, family: Family) -> Score:
        """Gets the likeliness of this actually being a family. Higher is more likely."""
        raise NotImplementedError()
