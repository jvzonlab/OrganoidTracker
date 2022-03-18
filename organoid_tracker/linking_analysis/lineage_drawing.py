# File originally written by Jeroen van Zon
from typing import Callable, Optional

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from organoid_tracker.core.links import LinkingTrack, Links
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import MPLColor

# Type definition: a color getter is a function that takes an int (time point) and the linking track, and returns a
# Matplotlib color
from organoid_tracker.gui.location_map import LocationMap

_ColorGetter = Callable[[int, LinkingTrack], MPLColor]
_LabelGetter = Callable[[LinkingTrack], Optional[str]]


def _get_lineage_drawing_start_time(lineage: LinkingTrack) -> int:
    """Gets the time at which the lineage should start to be drawn at the graph."""
    previous_tracks = lineage.get_previous_tracks()
    if previous_tracks:
        # Make lineage connect to previous lineage
        return previous_tracks.pop().max_time_point_number()
    return lineage.min_time_point_number()


def _no_filter(_track: LinkingTrack) -> bool:
    """Used as a default value in the lineage tree draw function. Makes all lineages show up."""
    return True


def _no_labels(_track: LinkingTrack) -> Optional[str]:
    """Used as a default value in the lineage tree draw function. Makes all lineages have no label."""
    return None


def _black(time_point_number: int, track: LinkingTrack) -> MPLColor:
    """Used as a default value in the lineage tree draw function. Makes all lineages black."""
    return 0, 0, 0


class LineageDrawing:
    links: Links

    def __init__(self, links: Links):
        self.links = links

    def _get_sublineage_draw_data(self, lin_id: LinkingTrack, x_curr_branch, x_end_branch, line_list):
        # if current branch has no next tracks
        if len(lin_id.get_next_tracks()) != 2:
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

    def draw_lineages_colored(self, axes: Axes, *, color_getter: _ColorGetter = _black,
                              resolution: ImageResolution = ImageResolution(1, 1, 1, 60),
                              location_map: LocationMap = LocationMap(),
                              label_getter: Callable[[LinkingTrack], Optional[str]] = _no_labels,
                              lineage_filter: Callable[[LinkingTrack], bool] = _no_filter,
                              line_width: float = 1.5):
        """Draws lineage trees that are color coded. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels."""

        x_offset = 0
        for lineage in self.links.find_starting_tracks():
            if not lineage_filter(lineage):
                continue
            width = self._draw_single_lineage_colored(axes, lineage, x_offset, color_getter, label_getter, resolution,
                                                      location_map, line_width)
            x_offset += width
        return x_offset

    def _draw_single_lineage_colored(self, ax: Axes, lineage: LinkingTrack, x_offset: int, color_getter: _ColorGetter,
                                     label_getter: _LabelGetter, image_resolution: ImageResolution,
                                     location_map: LocationMap, line_width: float) -> int:
        """Draw lineage with given function used for color. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels."""
        (diagram_width, line_list) = self._get_lineage_draw_data(lineage)

        lines_XY = []
        lines_col = []

        for line in line_list:
            # for current line, get timepoints T
            time_points_of_line = line[1]
            if len(time_points_of_line) == 2 and time_points_of_line[1] - time_points_of_line[0] >= 2:
                ### multiple timepoints T, so a vertical line
                # get x position of line
                X = line[0][0]
                # and cell id
                linking_track: LinkingTrack = line[2]

                # get indices of timepoints that bound the time interval T[0],T[1]
                time_point_min = time_points_of_line[0]
                time_point_max = time_points_of_line[1]

                color_val = color_getter(time_point_min + 1, linking_track)
                t0 = time_point_min * image_resolution.time_point_interval_h
                label = label_getter(linking_track)
                if label is not None:
                    ax.text(x_offset + X + 0.05, t0 + 0.4, label, verticalalignment='top', clip_on=True)
                for time_point_of_line in range(time_point_min, time_point_max):
                    # get time points for current sub time interval i

                    t1 = (time_point_of_line + 1) * image_resolution.time_point_interval_h
                    # get color corresponding to current z value
                    color_val_next = color_getter(time_point_of_line + 2, linking_track) if time_point_of_line + 2 <= time_point_max else None
                    if color_val_next != color_val:
                        # save line data
                        lines_XY.append([(x_offset + X, t0), (x_offset + X, t1)])
                        lines_col.append(color_val)

                        color_val = color_val_next
                        t0 = t1
                    location_map.set(int(x_offset + X), int(t1),
                                     linking_track.find_position_at_time_point_number(time_point_of_line + 1))
            if len(time_points_of_line) == 1:
                ### single timepoint T, so a horizontal line
                # get x position of line
                X = line[0]
                # and cell id
                linking_track = line[2]

                # get indeces of timepoint prior to T
                time_point_of_line = time_points_of_line[0]
                time = time_point_of_line * image_resolution.time_point_interval_h

                # get color corresponding to current z value
                color_val_next = color_getter(time_point_of_line, linking_track)
                # save line data
                lines_XY.append(
                    [(x_offset + X[0], time), (x_offset + X[1], time)])
                lines_col.append(color_val_next)
                location_map.set_area(int(x_offset + X[0]), int(time), int(x_offset + X[1]), int(time),
                                      linking_track.find_position_at_time_point_number(time_point_of_line))

        line_segments = LineCollection(lines_XY, colors=lines_col, lw=line_width)
        ax.add_collection(line_segments)

        return diagram_width

    def __repr__(self) -> str:
        return f"<Lineage tree of {self.links}>"
