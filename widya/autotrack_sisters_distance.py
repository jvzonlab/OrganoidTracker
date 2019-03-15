import math
from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
import matplotlib.pyplot as plt
import numpy as np


# Loading a new experiment from existing data
from widya.autotrack_get_symmetry import get_symmetry

experiment = io.load_data_file(
    "S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/31-1_correctdata.aut")

# Store mothers cell
mothers = cell_division_finder.find_mothers(experiment.links)

# Empty list for distance
symmetric_36_mins_list = []
symmetric_2h_list = []
asymmetric_36_mins_list = []
asymmetric_2h_list = []
count_1 = 0
count_2 = 0


# Get position and distance for every mother cells and their daughters
for mother in mothers:
    daughter1, daughter2 = experiment.links.find_futures(mother)
    distance_1 = daughter1.distance_squared(daughter2)
    distance_sqrt_1 = math.sqrt(distance_1)
    # Get position and distance of daughter cells if the distance less than  200*2, and only execute when it is 0 or more than 1
    while True:
        next_daughters1 = experiment.links.find_futures(daughter1)
        next_daughters2 = experiment.links.find_futures(daughter2)
        if len(next_daughters1) != 1 or len(next_daughters2) != 1:
          break
        daughter1 = next_daughters1.pop()
        daughter2 = next_daughters2.pop()
        distance_2 = daughter1.distance_squared(daughter2)
        distance_sqrt_2 = math.sqrt(distance_2)
        distance_um = experiment.images.resolution().pixel_size_x_um * distance_sqrt_2
        # Compare the distance of daughter cells in different time point
        if daughter1.time_point_number() == mother.time_point_number() + 30:
            track_1 = experiment.links.get_track(daughter1)
            track_2 = experiment.links.get_track(daughter2)
            print(len(track_1.get_next_tracks()), len(track_2.get_next_tracks()))
            if get_symmetry(experiment.links, track_1, track_2):
                symmetric_2h_list.append(distance_um)
                symmetric_36_mins_list.append(distance_36mins)
                count_1 = count_1+1
                print ("cells are symmetric")
            else:
                asymmetric_2h_list.append(distance_um)
                asymmetric_36_mins_list.append(distance_36mins)
                count_2 = count_2 + 1
                print(count_2, "cells are not symmetric")
        if daughter1.time_point_number() == mother.time_point_number() + 3:
            distance_36mins = distance_um

# .. Loop has ended, now our list is complete
# Make plot for distance comparison
plt.scatter(symmetric_2h_list, symmetric_36_mins_list, c = "blue")
plt.scatter(asymmetric_2h_list, asymmetric_36_mins_list, c ="yellow")
plt.xlabel('distance between sister cells nuclei after 2 hours')
plt.ylabel('distance between sister cells nuclei after 36 mins')
plt.suptitle('Sister cells distances after 2 hours v. 36 mins')
plt.show()
