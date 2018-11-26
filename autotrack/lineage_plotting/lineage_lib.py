# File originally written by Jeroen van Zon
from typing import List, Union, Callable, Any
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from autotrack.core import TimePoint
from autotrack.core.resolution import ImageResolution


class Lineage:

    # Lineage ids. Holds a single lineage_tree. A lineage_tree is either a cell_id (an int) or a list representing the
    # cell_id and the daughters it will divide into:
    # lineage_tree := cell_id   , if not dividing
    # lineage_tree := [cell_id, [daughter1, daughter2]]    , if dividing
    #
    # The daughters are then another lineage_tree. So if a daughter doesn't divide further it will simply be a cell_id,
    # otherwise a [cell_id, [granddaughter1, granddaughter2]].
    lineage_id: Union[int, List]
    n_cell: int  # Number of cells in the lineage == clonal size

    def __init__(self, lineage_id, lineage_interval, n_cell):
        self.lineage_id = lineage_id
        self.lin_interval = lineage_interval
        self.n_cell = n_cell

    def _lineage_divide_cell(self, lineage_ids, lin_interval, t_div, ind_mother, ind_daughter_list, n_cell):
        # if current branch is not a list
        if type(lineage_ids) != list:
            # then it has not sublineage
            if lineage_ids == ind_mother:
                # if the id of this branch is that of the dividing cell, implement the division
                # replace the id with a list of the two daughter cell ids
                lineage_ids = [ind_mother, ind_daughter_list]
                # replace the time of birth of cell <lin_id> with an interval
                # [t_birth_<lin_id>, [t_birth_daughter0, t_birth_daughter1]]
                lin_interval = [lin_interval, [t_div, t_div]]
                # and increase the cell count for the lineage
                n_cell = n_cell + 1
        else:
            # if it is a list, it has a sublineage
            for i in range(0, 2):
                # for each daughter sublineage, get the id and time interval data
                sub_lin_id = lineage_ids[1][i]
                sub_lin_interval = lin_interval[1][i]
                # and search for cell (and implement division when found) in sublineage
                (sub_lin_id, sub_lin_interval, n_cell) = self._lineage_divide_cell(sub_lin_id, sub_lin_interval, t_div,
                                                                                   ind_mother, ind_daughter_list, n_cell)
                # and update sublineages in lineage data
                lineage_ids[1][i] = sub_lin_id
                lin_interval[1][i] = sub_lin_interval

        # return updated lineage data        
        return lineage_ids, lin_interval, n_cell

    def _get_sublineage_draw_data(self, lin_id, lin_interval, t_end, x_curr_branch, x_end_branch, line_list):
        # if current branch is not a list
        if type(lin_id) != list:
            # then it has no sublineage, so we plot an end branch
            # set x position of current branch to that of the next end branch
            x_curr_branch = x_end_branch
            # plot line from time of birth to end time of lineage tree
            #            plt.plot([x_curr_branch,x_curr_branch],[lin_interval,t_end],'-k')
            X = [x_curr_branch]
            T = [lin_interval, t_end]
            CID = lin_id
            line_list.append([X, T, CID])
            #            plt.text(x_curr_branch, lin_interval, lin_id)
            # and increase the position of the next end branch 
            x_end_branch = x_end_branch + 1
        else:
            # if it is a list, it has a sublineage
            x = []
            for i in range(0, 2):
                # for each daughter sublineage, get the id and time interval data
                sub_lin_id = lin_id[1][i]
                sub_lin_interval = lin_interval[1][i]
                # and draw sublineage sublineage
                (x_curr_branch, x_end_branch, line_list) = self._get_sublineage_draw_data(sub_lin_id, sub_lin_interval,
                                                                                          t_end, x_curr_branch,
                                                                                          x_end_branch, line_list)
                # for each sublineage, save the current branch x position
                x.append(x_curr_branch)
            # get the start of the time interval
            t0 = lin_interval[0]
            CID = lin_id[0]
            # and the end            
            if type(lin_interval[1][0]) != list:
                t1 = lin_interval[1][0]
            else:
                t1 = lin_interval[1][0][0]
            # plot horizontal line connected the two daughter branches
            X = [x[0], x[1]]
            T = [t1]
            line_list.append([X, T, CID])

            # and plot the mother branch
            x_curr_branch = (x[0] + x[1]) / 2.
            X = [x_curr_branch]
            T = [t0, t1]
            line_list.append([X, T, CID])

        # return updated lineage data        
        return x_curr_branch, x_end_branch, line_list

    def get_lineage_draw_data(self, t_end: int):
        (x_curr, x_end, line_list) = self._get_sublineage_draw_data(self.lineage_id, self.lin_interval, t_end, 0, 0, [])
        return x_end, line_list

    def draw_lineage(self, axes: Axes, T_end: int, x_offset: int, show_cell_id: bool = False) -> int:
        """Draws a single lineage. Returns its width."""
        (diagram_width, line_list) = self.get_lineage_draw_data(T_end)

        for l in line_list:
            X = l[0]
            T = l[1]
            CID = l[2]
            if len(T) == 2:
                ## two timepoints T, so this is a vertical line
                # plot line
                axes.plot([x_offset + X[0], x_offset + X[0]], T, '-k')
                if show_cell_id:
                    # print cell id
                    plt.text(x_offset + X[0], T[0], CID)
            if len(X) == 2:
                ## two x positions, so this a horizontal line indicating division
                # plot line
                axes.plot([x_offset + X[0], x_offset + X[1]], [T[0], T[0]], '-k')
        return diagram_width

    def draw_lineage_diagram_with_data_as_color(self, ax: Axes, T_end, color_getter: Callable[[int, int], Any],
                                                data_color_range, x_offset, image_resolution: ImageResolution):
        """draw lineage with z position included as color.

        lineage_celldata: time points, cell ids (so that list will have a different length) and z_list by [cell index,
        time point index]
"""
        (diagram_width, line_list) = self.get_lineage_draw_data(T_end)

        lines_XY = []
        lines_col = []

        for line in line_list:
            # for current line, get timepoints T
            actual_times_of_line = line[1]
            if len(actual_times_of_line) == 2:
                ### multiple timepoints T, so a vertical line
                # get x position of line
                X = line[0][0]
                # and cell id
                cell_id = line[2]

                # get indeces of timepoints that bound the time interbal T[0],T[1]
                index_of_time_point_min = math.floor(actual_times_of_line[0] / image_resolution.time_point_interval_h)
                index_of_time_point_max = math.ceil(actual_times_of_line[1] / image_resolution.time_point_interval_h)

                for index_of_time_point in range(index_of_time_point_min, index_of_time_point_max):
                    # get time points for current sub time interval i
                    t0 = image_resolution.time_point_interval_h * index_of_time_point
                    t1 = image_resolution.time_point_interval_h * (index_of_time_point + 1)
                    # constrain bounds to interval T[0],T[1] if needed
                    if t0 < actual_times_of_line[0]:
                        t0 = actual_times_of_line[0]
                    if t1 > actual_times_of_line[1]:
                        t1 = actual_times_of_line[1]
                    # get data value
                    val = color_getter(index_of_time_point, cell_id)
                    # get color corresponding to current z value
                    color_val = cm.jet((val - data_color_range[0]) / (data_color_range[1] - data_color_range[0]))
                    # save line data
                    lines_XY.append([(x_offset + X, t0), (x_offset + X, t1)])
                    lines_col.append(color_val)
            if len(actual_times_of_line) == 1:
                ### single timepoint T, so a horizontal line
                # get x position of line
                X = line[0]
                # and cell id
                cell_id = line[2]

                # get indeces of timepoint prior to T
                index_of_time_point = math.floor(actual_times_of_line[0] / image_resolution.time_point_interval_h)

                # get z value
                val = color_getter(index_of_time_point, cell_id)
                # get color corresponding to current z value
                color_val = cm.jet((val - data_color_range[0]) / (data_color_range[1] - data_color_range[0]))
                # save line data
                lines_XY.append([(x_offset + X[0], actual_times_of_line[0]), (x_offset + X[1], actual_times_of_line[0])])
                lines_col.append(color_val)

        line_segments = LineCollection(lines_XY, colors=lines_col, lw=3)
        ax.add_collection(line_segments)

        return diagram_width

    def is_in_lineage(self, lin_id, cell_id):
        if type(lin_id) != list:
            if lin_id == cell_id:
                return True
        else:
            if lin_id[0] == cell_id:
                return True
            elif type(lin_id[1]) == list:
                for i in range(0, 2):
                    if self.is_in_lineage(lin_id[1][i], cell_id):
                        return True
            elif lin_id[1] == cell_id:
                return True

        return False

    def __repr__(self) -> str:
        return f"<Lineage, {self.n_cell} cells>"
