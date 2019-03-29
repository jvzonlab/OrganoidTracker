from autotrack.core.experiment import Experiment
from autotrack.core.links import LinkingTrack
from autotrack.linking_analysis.cell_fate_finder import get_fate, CellFateType


def get_symmetry(experiment: Experiment, track_1: LinkingTrack, track_2: LinkingTrack):
    fate_1 = get_fate(experiment, track_1.find_last_position())
    fate_2 = get_fate(experiment, track_2.find_last_position())
    if fate_1.type == CellFateType.WILL_DIVIDE and fate_2.type == CellFateType.WILL_DIVIDE:
        # Both are dividing
        return True
    if fate_1.type == CellFateType.WILL_DIVIDE or (fate_1.type == CellFateType.WILL_DIE or fate_1.type == CellFateType.WILL_SHED)\
            and fate_2.type == CellFateType.WILL_DIVIDE or (fate_2.type == CellFateType.WILL_DIE or fate_2.type == CellFateType.WILL_SHED):
        # One dead or shed, one divide
        return True
    if fate_1.type == CellFateType.JUST_MOVING and fate_2.type == CellFateType.JUST_MOVING:
        # Both are moving
        return True
    if fate_1.type == CellFateType.UNKNOWN or fate_2.type == CellFateType.UNKNOWN:
        # Don't know the fate of one of the cells
        return False
    return False
