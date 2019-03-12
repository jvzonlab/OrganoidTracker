import math

from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.core import TimePoint
import matplotlib.pyplot as plt
import numpy as np


# Loading a new experiment from existing data
experiment = io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/31-1_correctdata.aut")

# Store mothers cell
mothers = cell_division_finder.find_mothers(experiment.links)

# Empty list
distance_list = []
distance_36min_list = []
dz_0_list = []
dz_1_list = []
dz_2_list = []
dz_3_list = []
dz_4_list = []
dz_5_list = []
dz_6_list = []
dz_7_list = []



def get_symmetry(links, track_1, track_2):
    """Returns True if symmetric (both divide or both don't divide), False otherwise."""
    if len(track_1) == 0 or len(track_2) == 0:
        # cells are dead, so it's symmetric
        return True
    elif len(track_1) == 2 or len(track_2) == 2:
        # Cells are both dividing, so it's symmetric
        return True
    else:
        # One cell divides, other does not
        end_marker1 = linking_markers.get_track_end_marker(experiment.links, track_1.find_last_position())
        end_marker2 = linking_markers.get_track_end_marker(experiment.links, track_2.find_last_position())
        if end_marker1 == EndMarker.DEAD or end_marker2 == EndMarker.DEAD:
            # Cell died, other divided, so asymmetric
            return False
        # One of the cells went out of the view, so we can no longer track it
        # No idea if it's symmetric or not, but let's assume so
        return True


# Get position and distance for every mother cells and their daughters
for mother in mothers:
    daughter1, daughter2 = experiment.links.find_futures(mother)
    distance = daughter1.distance_squared(daughter2)
    distance_sqrt = math.sqrt(distance)
    distance_um = experiment.images.resolution().pixel_size_x_um * distance_sqrt
    dz_1 = daughter1.z - daughter2.z
    dz_2 = daughter2.z - daughter1.z
    if dz_1 == 1 or dz_2 == 1 :
        while True:
            next_daughters1 = experiment.links.find_futures(daughter1)
            next_daughters2 = experiment.links.find_futures(daughter2)
            if len(next_daughters1) != 1 or len(next_daughters2) != 1:
                break
            daughter1 = next_daughters1.pop()
            daughter2 = next_daughters2.pop()
            distance = daughter1.distance_squared(daughter2)
            # Compare the distance of daughter cells in different time point
            if daughter1.time_point_number() == mother.time_point_number() + 7:
                distance_list.append(distance_um)
                #count_1 = count_1 + 1
                print( distance_um)



plt.hist(distance_list)
plt.xlabel('Distance between the center of nucleus with z > 1 after division(μm)')
#plt.show()



