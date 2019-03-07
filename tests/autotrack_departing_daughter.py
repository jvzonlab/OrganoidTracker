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

# Empty list for distance
distance_list = []
distance_36min_list = []
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
symmetry =[]

def get_symmetry(links, track_1, track_2):
    """Returns True if symmetric (both divide or both don't divide), False otherwise."""
    if len(track_1) == 0 or len(track_2) == 0:
        # cells are dead, so it's symmetric
        return True
    elif len(track_1) == 2 or len(track_2)==2:
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


    while True:
        next_daughters1 = experiment.links.find_futures(daughter1)
        next_daughters2 = experiment.links.find_futures(daughter2)
        if len(next_daughters1) != 1 or len(next_daughters2) != 1:
            break
        daughter1 = next_daughters1.pop()
        daughter2 = next_daughters2.pop()
        distance = daughter1.distance_squared(daughter2)
        distance_sqrt = math.sqrt(distance)
        distance_um = experiment.images.resolution().pixel_size_x_um * distance_sqrt

        # Compare the distance of daughter cells in different time point
        if daughter1.time_point_number() == mother.time_point_number() + 7:
            distance_list.append(distance_um)
            distance_36min_list.append(distance_36mins)
            if distance_um > 11 or distance_36mins > 11 :
                count_1 = count_1 + 1
                print(count_1, 'mother', mother, 'daughter', daughter1, daughter2, 'with distance:', distance_um)
            else:



#plt.hist(distance_list, color = "lightgreen")
# .. Loop has ended, now our list is complete
# Make plot for distance comparison
#plt.plot(distance_sqrt, symmetry)
#plt.xlabel('Distance between the center of nucleus (μm)')
#plt.suptitle('The Distance Between Sister Cells After 2 hours')
plt.scatter(distance_list, distance_36min_list)
#plt.show()
