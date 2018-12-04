# File originally written by Jeroen van Zon
from typing import Callable, Tuple

from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from autotrack.core.links import LinkingTrack, ParticleLinks
from autotrack.core.resolution import ImageResolution

# Type definition: a color getter is a function that takes an int (time point) and the linking track, and returns a
# float value.
_ColorGetter = Callable[[int, LinkingTrack], float]


def _get_lineage_drawing_start_time(lineage: LinkingTrack) -> int:
    """Gets the time at which the lineage should start to be drawn at the graph."""
    previous_tracks = lineage.get_previous_tracks()
    if previous_tracks:
        # Make lineage connect to previous lineage
        return previous_tracks.pop().max_time_point_number()
    return lineage.min_time_point_number()


class LineageDrawing:

    links: ParticleLinks

    def __init__(self, links: ParticleLinks):
        self.links = links

    def _get_sublineage_draw_data(self, lin_id: LinkingTrack, x_curr_branch, x_end_branch, line_list):
        # if current branch has no next tracks
        if not lin_id.get_next_tracks():
            # then it has no sublineage, so we plot an end branch
            # set x position of current branch to that of the next end branch
            x_curr_branch = x_end_branch
            # plot line from time of birth to end time of lineage tree
            #            plt.plot([x_curr_branch,x_curr_branch],[lin_interval,t_end],'-k')
            X = [x_curr_branch]
            T = [_get_lineage_drawing_start_time(lin_id), lin_id.max_time_point_number()]
            CID = lin_id
            line_list.append([X, T, CID])
            #            plt.text(x_curr_branch, lin_interval, lin_id)
            # and increase the position of the next end branch 
            x_end_branch = x_end_branch + 1
        else:
            # it has sublineages
            x = []
            for sub_lin_id in lin_id.get_next_tracks():
                # for each daughter sublineage, get the time interval data
                # and draw sublineage sublineage
                (x_curr_branch, x_end_branch, line_list) = self._get_sublineage_draw_data(sub_lin_id, x_curr_branch,
                                                                                          x_end_branch, line_list)
                # for each sublineage, save the current branch x position
                x.append(x_curr_branch)
            # get the start of the time interval
            t0 = _get_lineage_drawing_start_time(lin_id)
            # and the end
            t1 = lin_id.max_time_point_number()
            # plot horizontal line connected the two daughter branches
            X = [x[0], x[1]]
            T = [t1]
            line_list.append([X, T, lin_id])

            # and plot the mother branch
            x_curr_branch = (x[0] + x[1]) / 2.
            X = [x_curr_branch]
            T = [t0, t1]
            line_list.append([X, T, lin_id])

        # return updated lineage data        
        return x_curr_branch, x_end_branch, line_list

    def _get_lineage_draw_data(self, lineage: LinkingTrack):
        (x_curr, x_end, line_list) = self._get_sublineage_draw_data(lineage, 0, 0, [])
        return x_end, line_list

    def draw_lineages(self, axes: Axes, show_cell_id: bool = False):
        """Draws a lineage tree of the links without color. Returns the width of the lineage in Matplotlib pixels."""
        x_offset = 0
        for lineage in self.links.find_starting_tracks():
            width = self._draw_single_lineage(axes, lineage, x_offset, show_cell_id)
            x_offset += width
        return x_offset

    def _draw_single_lineage(self, axes: Axes, lineage: LinkingTrack, x_offset: int, show_cell_id: bool) -> int:
        """Draws a single lineage. Returns the width of the lineage tree in Matplotlib pixels."""
        (diagram_width, line_list) = self._get_lineage_draw_data(lineage)

        for l in line_list:
            X = l[0]
            T = l[1]
            linking_track: LinkingTrack = l[2]
            if len(T) == 2:
                # two timepoints T, so this is a vertical line
                # plot line
                axes.plot([x_offset + X[0], x_offset + X[0]], T, '-k')
                if show_cell_id:
                    # print cell id
                    cell = linking_track.find_first()
                    axes.annotate(f"({cell.x:.1f}, {cell.y:.1f}, {cell.z:.1f})", (x_offset + X[0], T[0]))
            if len(X) == 2:
                # two x positions, so this a horizontal line indicating division
                # plot line
                axes.plot([x_offset + X[0], x_offset + X[1]], [T[0], T[0]], '-k')
        return diagram_width

    def draw_lineages_colored(self, axes: Axes, color_getter: _ColorGetter, data_color_range: Tuple[float, float],
                              image_resolution: ImageResolution):
        """Draws lineage trees that are color coded. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels."""
        x_offset = 0
        for lineage in self.links.find_starting_tracks():
            width = self._draw_single_lineage_colored(axes, lineage, x_offset, color_getter, data_color_range, image_resolution)
            x_offset += width
        return x_offset

    def _draw_single_lineage_colored(self, ax: Axes, lineage: LinkingTrack, x_offset: int, color_getter: _ColorGetter,
                                     data_color_range: Tuple[float, float], image_resolution: ImageResolution) -> int:
        """Draw lineage with given function used for color. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels."""
        (diagram_width, line_list) = self._get_lineage_draw_data(lineage)

        lines_XY = []
        lines_col = []

        for line in line_list:
            # for current line, get timepoints T
            time_points_of_line = line[1]
            if len(time_points_of_line) == 2:
                ### multiple timepoints T, so a vertical line
                # get x position of line
                X = line[0][0]
                # and cell id
                linking_track = line[2]

                # get indeces of timepoints that bound the time interbal T[0],T[1]
                time_point_min = time_points_of_line[0]
                time_point_max = time_points_of_line[1]

                for time_point_of_line in range(time_point_min, time_point_max):
                    # get time points for current sub time interval i
                    t0 = time_point_of_line * image_resolution.time_point_interval_h
                    t1 = (time_point_of_line + 1) * image_resolution.time_point_interval_h
                    # get data value
                    val = color_getter(time_point_of_line, linking_track)
                    # get color corresponding to current z value
                    color_val = cm.jet((val - data_color_range[0]) / (data_color_range[1] - data_color_range[0]))
                    # save line data
                    lines_XY.append([(x_offset + X, t0), (x_offset + X, t1)])
                    lines_col.append(color_val)
            if len(time_points_of_line) == 1:
                ### single timepoint T, so a horizontal line
                # get x position of line
                X = line[0]
                # and cell id
                linking_track = line[2]

                # get indeces of timepoint prior to T
                time_point_of_line = time_points_of_line[0]
                time = time_point_of_line * image_resolution.time_point_interval_h

                # get z value
                val = color_getter(time_point_of_line, linking_track)
                # get color corresponding to current z value
                color_val = cm.jet((val - data_color_range[0]) / (data_color_range[1] - data_color_range[0]))
                # save line data
                lines_XY.append(
                    [(x_offset + X[0], time), (x_offset + X[1], time)])
                lines_col.append(color_val)

        line_segments = LineCollection(lines_XY, colors=lines_col, lw=3)
        ax.add_collection(line_segments)

        return diagram_width

    def __repr__(self) -> str:
        return f"<Lineage tree of {self.links}>"
