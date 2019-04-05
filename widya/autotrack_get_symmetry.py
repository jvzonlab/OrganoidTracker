from autotrack.core.experiment import Experiment
from autotrack.core.links import LinkingTrack
from autotrack.linking_analysis.cell_fate_finder import get_fate, CellFateType


def get_symmetry(experiment: Experiment, track_1: LinkingTrack, track_2: LinkingTrack):
    fate_1 = get_fate(experiment, track_1.find_first_position())
    fate_2 = get_fate(experiment, track_2.find_first_position())
    if fate_1.type == CellFateType.WILL_DIVIDE:
        if fate_2.type == CellFateType.JUST_MOVING:
            return False
        if fate_2.type == CellFateType.UNKNOWN:
            print("UNKOWN didn't get filtered out - this is an error!")
            return True
        if fate_2.type == CellFateType.WILL_DIVIDE:
            return True
        if fate_2.type == CellFateType.WILL_SHED or fate_2.type == CellFateType.WILL_DIE:
            return True  # Maybe it would be symmetric if the sister didn't die
        print("Error: unkown cell type", fate_2.type)
        return True

    if fate_1.type == CellFateType.WILL_DIE or fate_1.type == CellFateType.WILL_SHED:
        return True

    if fate_1.type == CellFateType.UNKNOWN:
        print("UNKOWN didn't get filtered out - this is an error!")
        return True

    if fate_1.type == CellFateType.JUST_MOVING:
        if fate_2.type == CellFateType.JUST_MOVING:
            return True
        if fate_2.type == CellFateType.UNKNOWN:
            print("UNKOWN didn't get filtered out - this is an error!")
            return True
        if fate_2.type == CellFateType.WILL_DIVIDE:
            return False
        if fate_2.type == CellFateType.WILL_SHED or fate_2.type == CellFateType.WILL_DIE:
            return True  # Maybe it would be symmetric if the sister didn't die
        print("Error: unkown cell type", fate_2.type)
        return True

    print("Error: unkown cell type", fate_1.type)
    return True
