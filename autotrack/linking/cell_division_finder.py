"""Used to find divisions in linking data."""

import itertools
from typing import Set, List, Optional

from autotrack.core.image_loader import ImageLoader
from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position, PositionCollection
from autotrack.core.score import Family, ScoreCollection
from autotrack.linking.scoring_system import MotherScoringSystem


def get_next_division(links: PositionLinks, position: Position) -> Optional[Family]:
    """Gets the next division for the given position. Returns None if there is no such division. Raises ValueError if a
    cell with more than two daughters is found in this lineage."""
    track = links.get_track(position)
    if track is None:
        return None

    next_tracks = track.get_next_tracks()
    if len(next_tracks) == 0:
        return None

    next_daughters = [next_track.find_first_position() for next_track in next_tracks]
    if len(next_daughters) != 2:
        raise ValueError("Cell " + str(track.find_last_position()) + " has multiple daughters: " + str(next_daughters))

    return Family(track.find_last_position(), next_daughters[0], next_daughters[1])


def find_mothers(links: PositionLinks) -> Set[Position]:
    """Finds all mother cells in a graph. Mother cells are cells with at least two daughter cells."""
    mothers = set()

    for track in links.find_all_tracks():
        future_tracks = track.get_next_tracks()
        if len(future_tracks) >= 2:
            mothers.add(track.find_last_position())
        if len(future_tracks) > 2:
            print("Illegal mother: " + str(len(future_tracks)) + " daughters found")

    return mothers


def find_families(links: PositionLinks, warn_on_many_daughters = True) -> List[Family]:
    """Finds all mother and daughter cells in a graph. Mother cells are cells with at least two daughter cells.
    Returns a set of Family instances.
    """
    families = list()

    for track in links.find_all_tracks():
        next_tracks = track.get_next_tracks()
        if len(next_tracks) < 2:
            continue
        next_daughters = [next_track.find_first_position() for next_track in next_tracks]
        if warn_on_many_daughters:
            # Only two daughters are allowed
            families.append(Family(track.find_last_position(), next_daughters[0], next_daughters[1]))
            if len(next_tracks) > 2:
                print("Illegal mother: " + str(len(next_tracks)) + " daughters found")
        else:
            # Many daughters are allowed
            for daughter1, daughter2 in itertools.combinations(next_daughters, 2):
                families.append(Family(track.find_last_position(), daughter1, daughter2))

    return families


def calculates_scores(image_loader: ImageLoader, positions: PositionCollection, links: PositionLinks,
                      scoring_system: MotherScoringSystem) -> ScoreCollection:
    """Finds all families in the given links and calculates their scores."""
    scores = ScoreCollection()
    families = find_families(links, warn_on_many_daughters=False)
    i = 0
    for family in families:
        scores.set_family_score(family, scoring_system.calculate(image_loader, positions, family))
        i += 1
        if i % 100 == 0:
            print("   working on " + str(i) + "/" + str(len(families)) + "...")
    return scores