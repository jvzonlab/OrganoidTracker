from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
from autotrack.linking import nearby_position_finder
from autotrack.linking.nearby_position_finder import find_closest_n_positions
from autotrack.linking_analysis.cell_fate_finder import get_fate, CellFateType
from widya.autotrack_get_symmetry import get_symmetry
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
mother_density = []

symmetric_list = []
symmetric_axis_list = []
asymmetric_axis_list = []
asymmetric_list = []
count_1 = 0
count_2 = 0

for experiment_number, experiment in enumerate(experiments):
    # Store mothers cell
    mothers = cell_division_finder.find_mothers(experiment.links)

    # Get position for every mother cells and their daughters
    for mother in mothers:
        daughter1, daughter2 = experiment.links.find_futures(mother)
        fate_1 = get_fate(experiment, daughter1)
        fate_2 = get_fate(experiment, daughter2)
        if fate_1.type == CellFateType.UNKNOWN or fate_2.type == CellFateType.UNKNOWN:
            continue
        #if not (fate_1.type == CellFateType.WILL_DIVIDE or fate_2.type == CellFateType.WILL_DIVIDE):
           #continue  # Skip if none of the daughters divide
        # find fix number of closest cells to mother cells
        mother_position = experiment.positions.of_time_point(mother.time_point())
        mother_nearby_cells = find_closest_n_positions(mother_position, mother, 5)
        # list for all distance of 7 cells close to the mother
        nearby_mother_cell = []

        # get distance from each cells to mother cells
        for position in mother_nearby_cells:
            distance_nearby_cell_to_mother = position.distance_squared(mother)
            distance_sqrt_nearby_cell = math.sqrt(distance_nearby_cell_to_mother)
            distance_um_nearby_cells = experiment.images.resolution().pixel_size_x_um * distance_sqrt_nearby_cell
            nearby_mother_cell.append(distance_um_nearby_cells)

        # get average distance from 7 cells to mother cells
        average_distance_nearby_cells = np.average(nearby_mother_cell)
        average_nearby_cells_mother_um = experiment.images.resolution().pixel_size_x_um * average_distance_nearby_cells
        mother_density.append(average_nearby_cells_mother_um)
        #count_1 = count_1 + 1
        #print(count_1, average_mother_nearby)

        track_1 = experiment.links.get_track(daughter1)
        track_2 = experiment.links.get_track(daughter2)

        # check symmetry
        if get_symmetry(experiment, track_1, track_2):
            symmetric_list.append(average_nearby_cells_mother_um)
            count_1 = count_1 + 1
            #print(count_1, "cells are symmetric")
        else:
            asymmetric_list.append(average_nearby_cells_mother_um)
            count_2 = count_2 + 1
            print(count_2, experiment_number, mother, average_nearby_cells_mother_um, "cells are not symmetric")


plt.rcParams["font.family"] = "arial"

# Scott's rule histogram
stdv_mother = np.std(mother_density)
mean_mother = np.average(mother_density)
#print (mean_mother)
#print (stdv_mother)

n_mother = len(mother_density)
h_m_1 = (3.5*(stdv_mother))/(math.pow(n_mother, 1/3))

plt.hist(mother_density, bins=[h_m_1*15, h_m_1*16, h_m_1*17, h_m_1*18, h_m_1*19,h_m_1*20, h_m_1*21, h_m_1*22, h_m_1*23, h_m_1*24, h_m_1*25, h_m_1*26, h_m_1*27, h_m_1*28, h_m_1*29, h_m_1*30, h_m_1*31], color ='lightblue')

#symmetric mother plot
#plt.hist(symmetric_list,  bins=[h_m_1*15, h_m_1*16, h_m_1*17, h_m_1*18, h_m_1*19,h_m_1*20, h_m_1*21, h_m_1*22, h_m_1*23, h_m_1*24, h_m_1*25, h_m_1*26, h_m_1*27, h_m_1*28, h_m_1*29, h_m_1*30, h_m_1*31], color ='darksalmon')

#asymmetric mother plot
plt.hist(asymmetric_list, bins= [h_m_1*15, h_m_1*16, h_m_1*17, h_m_1*18, h_m_1*19,h_m_1*20, h_m_1*21, h_m_1*22, h_m_1*23, h_m_1*24, h_m_1*25, h_m_1*26, h_m_1*27, h_m_1*28, h_m_1*29, h_m_1*30, h_m_1*31], color ='red')

plt.xlabel('Neighbours cells average distance to mother cells (μm)')
plt.ylabel('Amount of cells with known fates')
plt.suptitle('Mother Cell Density')
plt.show()