from ai_track.imaging import io
from ai_track.linking import cell_division_finder
from ai_track.core.experiment import Experiment
from ai_track.core import TimePoint
from widya.ai_track_get_symmetry import get_symmetry
from ai_track.core.links import LinkingTrack
from ai_track.linking_analysis.cell_fate_finder import get_fate, CellFateType
import matplotlib.pyplot as plt
import numpy as np
import math




# Loading a new experiment from existing data
experiments =[
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/5-4_with_1_crypt.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-04-18_weekend_H2B-mCherry/nd410xy12-stacks/analyzed/with_axes(widya).aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-06-23_weekend_H2B-mCherry/nd478xy09-stacks/analyzed/with_axes_widya.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy16-stacks/analyzed/updated_axes_widya.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy08-stacks/analyzed/lineages.p")
]

# List
mother_pos_axis = []
asymmetric = []
symmetric = []
count_1 = 0
count_2 = 0

for experiment_number, experiment in enumerate(experiments):
    # Store mothers cell
    mothers = cell_division_finder.find_mothers(experiment.links)

    # Get position and distance for every mother cells and their daughters
    for mother in mothers:
        daughter1, daughter2 = experiment.links.find_futures(mother)
        # if one of the fate of one of the daughters is not known, continue
        fate_1 = get_fate(experiment, daughter1)
        fate_2 = get_fate(experiment, daughter2)
        if fate_1.type == CellFateType.UNKNOWN or fate_2.type == CellFateType.UNKNOWN:
            continue
        if (fate_1.type == CellFateType.WILL_DIE or fate_2.type == CellFateType.WILL_DIE) or \
                (fate_1.type == CellFateType.WILL_SHED or fate_2.type == CellFateType.WILL_SHED):
            continue

        # get the distance of mother to the axis
        pos_axis = experiment.splines.to_position_on_original_axis(experiment.links, mother).pos
        distance_um = experiment.images.resolution().pixel_size_x_um * pos_axis
        mother_pos_axis.append(distance_um)
        track_1 = experiment.links.get_track(daughter1)
        track_2 = experiment.links.get_track(daughter2)

        # check symmetry
        if get_symmetry(experiment, track_1, track_2):
            symmetric.append(distance_um)
            count_1 = count_1 + 1
            # exclude the unknown and dead cells
            print(count_1, "cells are symmetric")
        else:
            asymmetric.append(distance_um)
            count_2 = count_2 + 1
            print(count_2, experiment_number, mother, distance_um, "cells are not symmetric")

plt.rcParams["font.family"] = "arial"

# Scott's rule histogram
stdv_mother = np.std(mother_pos_axis)
mean_mother = np.average(mother_pos_axis)
#print (mean_mother)

n_mother = len(mother_pos_axis)
h_m_1 = (3.5*(stdv_mother))/(math.pow(n_mother, 1/3))

plt.hist(mother_pos_axis, bins=[0, h_m_1, h_m_1*2, h_m_1*3, h_m_1*4, h_m_1*5, h_m_1*6, h_m_1*7, h_m_1*8, h_m_1*9, h_m_1*10, h_m_1*11,h_m_1*12,h_m_1*13, h_m_1*14, h_m_1*15, h_m_1*16], color ='lightblue')

#symmetric mother plot
plt.hist(symmetric,  bins=[0, h_m_1, h_m_1*2, h_m_1*3, h_m_1*4, h_m_1*5, h_m_1*6, h_m_1*7, h_m_1*8, h_m_1*9, h_m_1*10, h_m_1*11,h_m_1*12,h_m_1*13, h_m_1*14, h_m_1*15, h_m_1*16], color ='darksalmon')

#asymmetric mother plot
plt.hist(asymmetric,  bins=[0, h_m_1, h_m_1*2, h_m_1*3, h_m_1*4, h_m_1*5, h_m_1*6, h_m_1*7, h_m_1*8, h_m_1*9, h_m_1*10, h_m_1*11,h_m_1*12,h_m_1*13, h_m_1*14, h_m_1*15, h_m_1*16], color ='red')

#Freedman–Diaconis' choice plot (change the std deviation to 2iQR)
iqr_mother = np.percentile(mother_pos_axis, 75, interpolation='higher') - np.percentile(mother_pos_axis, 25, interpolation='lower')

h_m_2 = (2*(iqr_mother))/(math.pow(n_mother, 1/3))

#plt.hist(mother_pos_axis, bins=[0, h_m_2, h_m_2*2, h_m_2*3, h_m_2*4, h_m_2*5, h_m_2*6, h_m_2*7, h_m_2*8, h_m_2*9, h_m_2*10, h_m_2*11,h_m_2*12,h_m_2*13, h_m_2*14, h_m_2*15, h_m_2*16,h_m_2*17], color ='lightblue')

#asymmetric mother plot
#plt.hist(asymmetric, bins=[0, h_m_2, h_m_2*2, h_m_2*3, h_m_2*4, h_m_2*5, h_m_2*6, h_m_2*7, h_m_2*8, h_m_2*9, h_m_2*10, h_m_2*11,h_m_2*12,h_m_2*13, h_m_2*14, h_m_2*15, h_m_2*16, h_m_2*17], color ='red')


plt.xlabel('Position in crypt-villus axis(μm)')
plt.ylabel('Amount of cells with known fates')
plt.suptitle('Mother Cells Positions in Crypt-Villus Axis')
plt.show()
