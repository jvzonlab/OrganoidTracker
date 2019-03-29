from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
from autotrack.linking import nearby_position_finder
from autotrack.linking.nearby_position_finder import find_closest_n_positions
from widya.autotrack_get_symmetry import get_symmetry
import matplotlib.pyplot as plt
import numpy as np
import math



# Loading a new experiment from existing data
experiments =[
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/15-3_with_axis.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-04-18_weekend_H2B-mCherry/nd410xy12-stacks/analyzed/with_axes(widya).aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-06-23_weekend_H2B-mCherry/nd478xy09-stacks/analyzed/with_axes_widya.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy16-stacks/analyzed/updated_axes_widya.aut"),
    io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy08-stacks/analyzed/lineages.p")
]

mother_pos_axis = []

for experiment in experiments:
    # Store mothers cell
    mothers = cell_division_finder.find_mothers(experiment.links)

    # Get position for every mother cells and their daughters
    for mother in mothers:
        # get the distance of mother to the axis
        pos_axis = experiment.data_axes.to_position_on_original_axis(experiment.links, mother).pos
        distance_um = experiment.images.resolution().pixel_size_x_um * pos_axis
        mother_pos_axis.append(distance_um)

        # find fix number of closest cells to mother cells
        mother_position = experiment.positions.of_time_point(mother.time_point())
        mother_nearby_cells = find_closest_n_positions(mother_position, mother, 7)
        # list for all distance of 7 cells close to the mother
        nearby_mother_cell = []
        average_mother_nearby = []

        # get distance from each cells to mother cells
        for position in mother_nearby_cells:
            distance_nearby_cell_to_mother = position.distance_squared(mother)
            nearby_mother_cell.append(distance_nearby_cell_to_mother)
            distance_sqrt_nearby_cell = math.sqrt(distance_nearby_cell_to_mother)
            distance_um_nearby_cells = experiment.images.resolution().pixel_size_x_um * distance_sqrt_nearby_cell
        # get average distance from 7 cells to mother cells
        average_distance_nearby_cells = np.average(nearby_mother_cell)
        average_nearby_cells_mother_um = experiment.images.resolution().pixel_size_x_um * average_distance_nearby_cells
        average_mother_nearby.append(average_nearby_cells_mother_um)
        #count_1 = count_1 + 1
        #print(count_1, average_mother_nearby)

        plt.rcParams["font.family"] = "arial"
        plt.plot(average_mother_nearby, color ='lightblue')
        plt.xlabel('Distance to crypt axis')
        plt.ylabel('Cell density (μm)')
        plt.suptitle('Mother Cells Positions in Crypt-Villus Axis')
        plt.show()