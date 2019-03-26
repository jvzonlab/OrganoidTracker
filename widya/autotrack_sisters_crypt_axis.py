from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.core import TimePoint
import matplotlib.pyplot as plt
import numpy as np
import math
from widya.autotrack_get_symmetry import get_symmetry

# List
mother_pos_axis = []
asymmetric = []
#sister1_pos_axis =[]
#sister2_pos_axis = []
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

for experiment in experiments:
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
                if daughter1.time_point_number() == mother.time_point_number() + 10:
                    # get the distance of sister cells to the axis
                    pos_axis_d1 = experiment.data_axes.to_position_on_original_axis(experiment.links, daughter1).pos
                    pos_axis_d2 = experiment.data_axes.to_position_on_original_axis(experiment.links, daughter2).pos
                    distance_um_d1 = experiment.images.resolution().pixel_size_x_um * pos_axis_d1
                    distance_um_d2 = experiment.images.resolution().pixel_size_x_um * pos_axis_d2
                    #sister1_pos_axis.append(distance_um_d1)
                    #sister2_pos_axis.append(distance_um_d2)
                    d_pos = abs(distance_um_d1 - distance_um_d2)
                    sisters_distance_list.append(d_pos)
                    #print(d_pos)
                    track_1 = experiment.links.get_track(daughter1)
                    track_2 = experiment.links.get_track(daughter2)
                    # check symmetry
                    if get_symmetry(experiment.links, track_1, track_2):
                        count_1 = count_1 + 1
                        #print(count_1, "cells are symmetric")
                    else:
                        asymmetric.append(d_pos)
                        count_2 = count_2 + 1
                        #print(count_2, mother, distance_um_2, "cells are not symmetric")
                        #print(count_2, mother, d_pos, "cells are not symmetric")

plt.rcParams["font.family"] = "arial"

# Scott's rule
stdv_mother = np.std(mother_pos_axis)
stdv_sister = np.std(sisters_distance_list)
stdv_asym = np.std(asymmetric)
n_mother = len(mother_pos_axis)
n_asym = len(asymmetric)
n_sister = len(sisters_distance_list)
h_m = (3.5*(stdv_mother))/(math.pow(n_mother, 1/3))
h_a = (3.5*(stdv_asym))/(math.pow(n_asym, 1/3))
h_s = (3.5*(stdv_sister))/(math.pow(n_sister, 1/3))
#print(h_a)
#plt.hist(mother_pos_axis, bins=[0, h_m, h_m*2, h_m*3, h_m*4, h_m*5, h_m*6, h_m*7, h_m*8, h_m*9, h_m*10, h_m*11,h_m*12,h_m*13, h_m*14, h_m*15, h_m*16], color ='lightblue')

#plt.hist(mother_pos_axis, bins = [0,10,20,30,40,50,60,70,80,90], color ='lightblue')
plt.hist(sisters_distance_list,bins =[0, h_s, h_s*2, h_s*3, h_s*4, h_s*5, h_s*6, h_s*7, h_s*8, h_s*9, h_s*10, h_s*11,h_s*12, h_s*13, h_s*14], color ='lightblue')

#asymmetric mother plot
#plt.hist(asymmetric, bins=[0, h_a, h_a*2, h_a*3, h_a*4, h_a*5, h_a*6, h_a*7, h_a*8, h_a*9, h_a*10], color ='red')

#asymmetric sister plot
plt.hist(asymmetric, bins=[0, h_a, h_a*2, h_a*3, h_a*4], color ='red')

#plt.xlabel('Position in crypt-villus axis(μm)')
plt.xlabel('Differences in position in crypt-villus axis (μm)')
plt.ylabel('Amount of cells')
plt.suptitle('Differences in Position in Cyrpt-Villus Axis Between Two Sister Cells')
#plt.suptitle('Mother Cells Positions in Crypt-Villus Axis')
plt.show()