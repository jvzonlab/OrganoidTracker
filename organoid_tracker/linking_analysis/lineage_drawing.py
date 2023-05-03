# File originally written by Jeroen van Zon
from typing import Callable, Optional, List, Union, NamedTuple, Tuple

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


class _Line(NamedTuple):
    """Holds either a horizontal line (t_start == t_end) or a vertical line (x_start == x_end), but never both."""
    time_point_number_start: int
    time_point_number_end: int
    track: LinkingTrack
    x_start: float
    x_end: float

    @staticmethod
    def vertical(*, x: float, track: LinkingTrack) -> "_Line":
        t_start = _get_lineage_drawing_start_time(track)
        t_end = track.max_time_point_number()
        return _Line(time_point_number_start=t_start, time_point_number_end=t_end, x_start=x, x_end=x, track=track)

    @staticmethod
    def horizontal(*, x_start: float, x_end: float, track: LinkingTrack) -> "_Line":
        time_point_number = track.max_time_point_number()
        return _Line(time_point_number_start=time_point_number, time_point_number_end=time_point_number,
                     x_start=x_start, x_end=x_end, track=track)

    def is_vertical(self) -> bool:
        return self.x_start == self.x_end

    def is_horizontal(self) -> bool:
        return self.x_start != self.x_end


class LineageDrawing:
    starting_tracks: List[LinkingTrack]

    def __init__(self, links: Union[Links, List[LinkingTrack]]):
        if isinstance(links, Links):
            self.starting_tracks = list(links.find_starting_tracks())
        else:
            self.starting_tracks = links

    def _get_sublineage_draw_data(self, linking_track: LinkingTrack, x_curr_branch: float, x_end_branch: float,
                                  line_list: List[_Line]) -> Tuple[float, float, List[_Line]]:
        next_tracks = linking_track.get_next_tracks()
        if len(next_tracks) == 0:
            # Plot an end branch
            x_curr_branch = x_end_branch
            line_list.append(_Line.vertical(x=x_curr_branch, track=linking_track))
            x_end_branch = x_end_branch + 1
        else:
            # Plot sublineages
            x_daughters = []
            for daughter_track in next_tracks:
                if daughter_track.get_previous_tracks().pop() != linking_track:
                    continue  # Ignore this one, it will be drawn by another parent
                (x_curr_branch, x_end_branch, line_list) = self._get_sublineage_draw_data(daughter_track, x_curr_branch,
                                                                                          x_end_branch, line_list)
                x_daughters.append(x_curr_branch)

            # Plot horizontal line connecting the daughter branches
            # (but don't do this if we only have one next track)
            if len(x_daughters) > 1:
                line_list.append(_Line.horizontal(x_start=x_daughters[0], x_end=x_daughters[-1], track=linking_track))

            # Plot the mother branch
            x_curr_branch = (x_daughters[0] + x_daughters[-1]) / 2 if len(x_daughters) > 0 else x_end_branch
            if len(x_daughters) == 0:
                # Found an end branch, make some room for it
                x_end_branch = x_end_branch + 1
            line_list.append(_Line.vertical(x=x_curr_branch, track=linking_track))

        return x_curr_branch, x_end_branch, line_list

    def _get_lineage_draw_data(self, lineage: LinkingTrack) -> Tuple[float, List[_Line]]:
        (x_curr, x_end, line_list) = self._get_sublineage_draw_data(lineage, 0, 0, [])
        return x_end, line_list

    def draw_lineages_colored(self, axes: Axes, *, color_getter: _ColorGetter = _black,
                              resolution: ImageResolution = ImageResolution(1, 1, 1, 60),
                              location_map: LocationMap = LocationMap(),
                              label_getter: Callable[[LinkingTrack], Optional[str]] = _no_labels,
                              lineage_filter: Callable[[LinkingTrack], bool] = _no_filter,
                              line_width: float = 1.5, x_offset_start: float = 0):
        """Draws lineage trees that are color coded. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels."""

        x_offset = x_offset_start
        for lineage in self.starting_tracks:
            if not lineage_filter(lineage):
                continue
            width = self._draw_single_lineage_colored(axes, lineage, x_offset, color_getter, label_getter, resolution,
                                                      location_map, line_width)
            x_offset += width
        return x_offset - x_offset_start

    def _draw_single_lineage_colored(self, ax: Axes, lineage: LinkingTrack, x_offset: float, color_getter: _ColorGetter,
                                     label_getter: _LabelGetter, image_resolution: ImageResolution,
                                     location_map: LocationMap, line_width: float) -> float:
        """Draw lineage with given function used for color. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels."""
        (diagram_width, line_list) = self._get_lineage_draw_data(lineage)

        lines_XY = []
        lines_col = []

        for line in line_list:
            # for current line, get timepoints T
            time_points_of_line = line[1]
            if line.is_vertical() and line.time_point_number_end - line.time_point_number_start >= 2:
                ### multiple timepoints T, so a vertical line
                # get x position of line
                x = line.x_start
                # and cell id
                linking_track: LinkingTrack = line[2]

                # get indices of timepoints that bound the time interval T[0],T[1]
                time_point_min = line.time_point_number_start
                time_point_max = line.time_point_number_end

                color_val = color_getter(time_point_min + 1, linking_track)
                t0 = time_point_min * image_resolution.time_point_interval_h
                label = label_getter(linking_track)
                if label is not None:
                    ax.text(x_offset + x + 0.05, t0 + 0.4, label, verticalalignment='top', clip_on=True)
                for time_point_of_line in range(time_point_min, time_point_max):
                    # get time points for current sub time interval i

                    t1 = (time_point_of_line + 1) * image_resolution.time_point_interval_h
                    # get color corresponding to current z value
                    color_val_next = color_getter(time_point_of_line + 2, linking_track) if time_point_of_line + 2 <= time_point_max else None
                    if color_val_next != color_val:
                        # save line data
                        lines_XY.append([(x_offset + x, t0), (x_offset + x, t1)])
                        lines_col.append(color_val)

                        color_val = color_val_next
                        t0 = t1
                    location_map.set(int(x_offset + x), int(t1),
                                     linking_track.find_position_at_time_point_number(time_point_of_line + 1))
            if line.is_horizontal():
                linking_track = line.track

                # get indeces of timepoint prior to T
                time_point_of_line = line.time_point_number_start
                time = time_point_of_line * image_resolution.time_point_interval_h

                # get color corresponding to current z value
                color_val_next = color_getter(time_point_of_line, linking_track)
                # save line data
                lines_XY.append(
                    [(x_offset + line.x_start, time), (x_offset + line.x_end, time)])
                lines_col.append(color_val_next)
                location_map.set_area(int(x_offset + line.x_start), int(time), int(x_offset + line.x_end), int(time),
                                      linking_track.find_position_at_time_point_number(time_point_of_line))

        line_segments = LineCollection(lines_XY, colors=lines_col, lw=line_width)
        ax.add_collection(line_segments)

        return diagram_width

    def __repr__(self) -> str:
        return f"<Lineage tree of {len(self.starting_tracks)} cells>"
