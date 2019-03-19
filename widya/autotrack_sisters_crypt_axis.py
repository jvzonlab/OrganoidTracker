from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.core import TimePoint
from widya.autotrack_get_symmetry import get_symmetry
import matplotlib.pyplot as plt
import numpy as np
import math

# List
mother_pos_axis = []
sister1_pos_axis =[]
sister2_pos_axis = []
count_1 = 0
count_2 = 0


# Loading a new experiment from existing data
experiment = io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/15-3_with_axis.aut")

# Store mothers cell
mothers = cell_division_finder.find_mothers(experiment.links)

# Get position and distance for every mother cells and their daughters
for mother in mothers:
    daughter1, daughter2 = experiment.links.find_futures(mother)
    distance_1 = daughter1.distance_squared(daughter2)
    distance_sqrt_1 = math.sqrt(distance_1)
    distance_um_1 = experiment.images.resolution().pixel_size_x_um * distance_sqrt_1
    # get the distance of mother to the axis
    pos_axis = experiment.data_axes.to_position_on_original_axis(experiment.links, mother).pos
    distance_um_2 = experiment.images.resolution().pixel_size_x_um * pos_axis
    mother_pos_axis.append(distance_um_2)
    #print(mother, distance_um_2)
    while True:
            next_daughters1 = experiment.links.find_futures(daughter1)
            next_daughters2 = experiment.links.find_futures(daughter2)
            if len(next_daughters1) != 1 or len(next_daughters2) != 1:
                break
            daughter1 = next_daughters1.pop()
            daughter2 = next_daughters2.pop()
            # get the distance of sister cells to the axis
            pos_axis_d1 = experiment.data_axes.to_position_on_original_axis(experiment.links, daughter1).pos
            pos_axis_d2 = experiment.data_axes.to_position_on_original_axis(experiment.links, daughter2).pos
            distance_um_d1 = experiment.images.resolution().pixel_size_x_um * pos_axis_d1
            distance_um_d2 = experiment.images.resolution().pixel_size_x_um * pos_axis_d2
            distance_d1_d2 = distance_um_d1 or distance_um_d2
            if daughter1.time_point_number() == mother.time_point_number() + 10:
                sister1_pos_axis.append(distance_um_d1)
                sister2_pos_axis.append(distance_um_d2)
                if distance_d1_d2 > 40:
                    track_1 = experiment.links.get_track(daughter1)
                    track_2 = experiment.links.get_track(daughter2)
                # check symmetry
                    if get_symmetry(experiment.links, track_1, track_2):
                        count_1 = count_1 + 1
                        print(count_1, "cells are symmetric")
                    else:
                        count_2 = count_2 + 1
                        print(count_2, mother, daughter1, daughter2, "cells are not symmetric")


plt.hist(sister1_pos_axis, color ='lightblue')
#plt.hist(sister2_pos_axis, color ='blue')
#plt.hist(mother_pos_axis, color ='lightgreen')
plt.xlabel('Distance (um)')
plt.suptitle('Sister cell distance to axis')
#lt.suptitle('Mother cells distance to axis')
plt.show()
