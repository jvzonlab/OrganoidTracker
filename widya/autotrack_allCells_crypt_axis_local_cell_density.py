from ai_track.imaging import io
from ai_track.linking import cell_division_finder
from ai_track.core.experiment import Experiment
from ai_track.core import TimePoint
from ai_track.linking import nearby_position_finder
from ai_track.linking.nearby_position_finder import find_closest_n_positions
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

# List
cell_pos_axis = []
cells_density = []


for experiment_number, experiment in enumerate(experiments):
    for cell in experiment.positions:
        # get the distance of all cells to the axis
        pos_axis = experiment.splines.to_position_on_original_axis(experiment.links, cell).pos
        distance_um = experiment.images.resolution().pixel_size_x_um * pos_axis
        cell_pos_axis.append(distance_um)

        # find fix number of closest cells to all cells
        cells_position = experiment.positions.of_time_point(cell.time_point())
        cells_nearby_cells = find_closest_n_positions(cells_position, cell, 5)
        # list for all distance of 7 cells close to the mother
        nearby_cells = []

        # get distance from each cells to all cells
        for position in cells_nearby_cells:
            distance_nearby_cell = position.distance_squared(cell)
            distance_sqrt_nearby_cell = math.sqrt(distance_nearby_cell)
            distance_um_nearby_cells = experiment.images.resolution().pixel_size_x_um * distance_sqrt_nearby_cell
            nearby_cells.append(distance_um_nearby_cells)

        # get average distance from 5 cells to mother cells
        average_distance_nearby_cells = np.average(nearby_cells)
        average_nearby_cells_um = experiment.images.resolution().pixel_size_x_um * average_distance_nearby_cells
        cells_density.append(average_nearby_cells_um)


# Plot for asymmteric and symmetric from mother cells distance to crypt villus axis and cell density
plt.rcParams["font.family"] = "arial"
plt.scatter(cell_pos_axis, cells_density, color ='lightblue')
plt.xlabel('Cells position in crypt-villus axis(μm)')
plt.ylabel('Neighbours cells average distance to cells (μm (μm)')
plt.suptitle('Cells Positions in Crypt-Villus Axis and Local Cell Density')
plt.show()
