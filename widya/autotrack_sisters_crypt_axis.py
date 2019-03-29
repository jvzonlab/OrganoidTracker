from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
from widya.autotrack_get_symmetry import get_symmetry
import matplotlib.pyplot as plt
import numpy as np
import math


# List
asymmetric = []
symmetric = []
sisters_distance_list =[]
count_1 = 0
count_2 = 0


# Loading a new experiment from existing data
experiments =[
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/15-3_with_axis.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-04-18_weekend_H2B-mCherry/nd410xy12-stacks/analyzed/with_axes(widya).aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-06-23_weekend_H2B-mCherry/nd478xy09-stacks/analyzed/with_axes_widya.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy16-stacks/analyzed/updated_axes_widya.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy08-stacks/analyzed/lineages.p")
]

for experiment_number, experiment in enumerate(experiments):
    # Store mothers cell
    mothers = cell_division_finder.find_mothers(experiment.links)

    # Get position and distance for every mother cells and their daughters
    for mother in mothers:
        daughter1, daughter2 = experiment.links.find_futures(mother)

        # get the distance of sister cells to the axis
        pos_axis_d1 = experiment.data_axes.to_position_on_original_axis(experiment.links, daughter1).pos
        pos_axis_d2 = experiment.data_axes.to_position_on_original_axis(experiment.links, daughter2).pos
        distance_um_d1 = experiment.images.resolution().pixel_size_x_um * pos_axis_d1
        distance_um_d2 = experiment.images.resolution().pixel_size_x_um * pos_axis_d2
        d_pos = abs(distance_um_d1 - distance_um_d2)
        sisters_distance_list.append(d_pos)
        track_1 = experiment.links.get_track(daughter1)
        track_2 = experiment.links.get_track(daughter2)

        # check symmetry
        if get_symmetry(experiment, track_1, track_2):
            symmetric.append(d_pos)
            count_1 = count_1 + 1
            print(count_1, "cells are symmetric")
        else:
            asymmetric.append(d_pos)
            count_2 = count_2 + 1
            print(count_2, mother, d_pos, "cells are not symmetric")


plt.rcParams["font.family"] = "arial"

# Scott's rule histogram
stdv_sister = np.std(sisters_distance_list)
mean_sister = np.average(sisters_distance_list)

#print (mean_sister)

n_sister = len(sisters_distance_list)
h_s_1 = (3.5*(stdv_sister))/(math.pow(n_sister, 1/3))

plt.hist(sisters_distance_list,bins =[0, h_s_1, h_s_1*2, h_s_1*3, h_s_1*4, h_s_1*5, h_s_1*6, h_s_1*7, h_s_1*8, h_s_1*9, h_s_1*10, h_s_1*11,h_s_1*12, h_s_1*13, h_s_1*14], color ='lightblue')

#symmetric sister plot
#plt.hist(symmetric, bins =[0, h_s_1, h_s_1*2, h_s_1*3, h_s_1*4, h_s_1*5, h_s_1*6, h_s_1*7, h_s_1*8, h_s_1*9, h_s_1*10, h_s_1*11,h_s_1*12, h_s_1*13, h_s_1*14], color ='darksalmon')


#asymmetric sister plot
plt.hist(asymmetric, bins =[0, h_s_1, h_s_1*2, h_s_1*3, h_s_1*4, h_s_1*5, h_s_1*6, h_s_1*7, h_s_1*8, h_s_1*9, h_s_1*10, h_s_1*11,h_s_1*12, h_s_1*13, h_s_1*14], color ='red')

#Freedman–Diaconis' choice plot (change the std deviation to 2iQR)
iqr_sisters = np.percentile(sisters_distance_list, 75, interpolation='higher') - np.percentile(sisters_distance_list, 25, interpolation='lower')
h_s_2= (2*(iqr_sisters))/(math.pow(n_sister, 1/3))

#plt.hist(sisters_distance_list,bins =[0, h_s_2, h_s_2*2, h_s_2*3, h_s_2*4, h_s_2*5, h_s_2*6, h_s_2*7, h_s_2*8, h_s_2*9, h_s_2*10, h_s_2*11,h_s_2*12, h_s_2*13, h_s_2*14, h_s_2*15], color ='lightblue')


#asymmetric sister plot
#plt.hist(asymmetric, bins =[0, h_s_2, h_s_2*2, h_s_2*3, h_s_2*4, h_s_2*5, h_s_2*6, h_s_2*7, h_s_2*8, h_s_2*9, h_s_2*10, h_s_2*11,h_s_2*12, h_s_2*13, h_s_2*14, h_s_2*15], color ='red')


plt.xlabel('Differences in position in crypt-villus axis (μm)')
plt.ylabel('Amount of cells')
plt.suptitle('Differences in Position in Cyrpt-Villus Axis Between Two Sister Cells')
plt.show()
