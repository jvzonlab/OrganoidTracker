# File originally written by Jeroen van Zon
from typing import Callable, Optional, List, Union, NamedTuple, Tuple, Set

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from organoid_tracker.core.links import LinkingTrack, Links
from organoid_tracker.core.resolution import ImageResolution, ImageTimings
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui.location_map import LocationMap

# Type definition: a color getter is a function that takes an int (time point) and the linking track, and returns a
# Matplotlib color
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


def _get_vertical_lines(x: float, linking_track: LinkingTrack, next_tracks: Set[LinkingTrack]) -> List[_Line]:
    """Draws the vertical line for the track, along with any cells that will merge into this track.

    In general, our code cannot draw cell nucleus merges. That would be extremely difficult to code. However,
    we make an exception for the case where all but one of the merging cells are "simple": they consist of only a
    single vertical line, with nothing prior."""

    # Find what tracks to draw
    tracks_to_draw = [linking_track]
    if len(next_tracks) == 1:
        # A possible cell merge
        next_track = next(iter(next_tracks))
        previous_tracks = linking_track.get_previous_tracks()
        potential_multiline_tracks = next_track.get_previous_tracks()
        if len(potential_multiline_tracks) >= 2:
            # Yes, a cell merge. Now check if we draw it as a double (or more) line
            i = 1
            for potential_multiline_track in potential_multiline_tracks:
                if potential_multiline_track is linking_track:
                    continue  # Self, ignore
                that_previous_tracks = potential_multiline_track.get_previous_tracks()
                if len(that_previous_tracks) > 0 and that_previous_tracks != previous_tracks:
                    return []  # Can't draw multilines, situation is too complex
                tracks_to_draw.append(potential_multiline_track)
                i += 1

    # Decide where vertical lines are going to be
    lines_list = list()
    x_start = x - (len(tracks_to_draw) - 1) / 4
    x_end = x + (len(tracks_to_draw) - 1) / 4
    for i, track_to_draw in enumerate(tracks_to_draw):
        lines_list.append(_Line.vertical(x=x_start + i / 2, track=track_to_draw))
    if x_end > x_start:
        # Add horizontal line connecting the lines
        lines_list.append(_Line.horizontal(x_start=x_start, x_end=x_end, track=linking_track))

    return lines_list


def _get_timings(resolution: Optional[ImageResolution], timings: Optional[ImageTimings]) -> Tuple[ImageTimings, bool]:
    """Gets the timings for the experiment. If no timings are available, then a time resolution of 1 hour per time point
    is assumed, and False is returned as well. This allows you to plot the timing as raw time points."""
    if resolution is not None and timings is not None:
        raise ValueError("Cannot specify both a resolution and specific timings")
    if timings is not None:
        return timings, True
    if resolution is not None:
        return ImageTimings.contant_timing(resolution.time_point_interval_m), True
    return ImageTimings.contant_timing(60), False


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
            return x_curr_branch, x_end_branch, line_list

        # Plot sublineages
        x_next_tracks = []
        for next_track in next_tracks:
            if next_track.get_previous_tracks().pop() != linking_track:
                continue  # Ignore this one, it will be drawn by another parent
            (x_curr_branch, x_end_branch, line_list) = self._get_sublineage_draw_data(next_track, x_curr_branch,
                                                                                      x_end_branch, line_list)
            x_next_tracks.append(x_curr_branch)

        # Plot horizontal line connecting the daughter branches
        # (but don't do this if we only have one next track)
        if len(x_next_tracks) > 1:
            line_list.append(_Line.horizontal(x_start=x_next_tracks[0], x_end=x_next_tracks[-1], track=linking_track))

        # Plot the mother branch
        x_curr_branch = (x_next_tracks[0] + x_next_tracks[-1]) / 2 if len(x_next_tracks) > 0 else x_end_branch

        # Draw our own lines
        main_lines = _get_vertical_lines(x_curr_branch, linking_track, next_tracks)
        if len(main_lines) <= 1:
            # A simple, vertical line - reserve space for it
            line_list += main_lines
            x_end_branch = x_end_branch + 1
        elif len(x_next_tracks) > 0:
            # A vertical line that will connect to something else
            line_list += main_lines
        # else: we have multiple lines to draw for this track - this indices that multiple cells will
        # merge into this one. We don't draw it though, since some other track was responsible for
        # drawing the next track(s), so that one should be the one to draw the multilines

        return x_curr_branch, x_end_branch, line_list

    def _get_lineage_draw_data(self, lineage: LinkingTrack) -> Tuple[float, List[_Line]]:
        (x_curr, x_end, line_list) = self._get_sublineage_draw_data(lineage, 0, 0, [])
        return x_end, line_list

    def draw_lineages_colored(self, axes: Axes, *, color_getter: _ColorGetter = _black,
                              resolution: Optional[ImageResolution] = None,
                              timings: Optional[ImageTimings] = None,
                              location_map: LocationMap = LocationMap(),
                              label_getter: Callable[[LinkingTrack], Optional[str]] = _no_labels,
                              lineage_filter: Callable[[LinkingTrack], bool] = _no_filter,
                              line_width: float = 1.5, x_offset_start: float = 0,
                              set_ylabel: bool = True):
        """Draws lineage trees that are color coded. You can for example color cells by z position, by track
        length, etc. Returns the width of the lineage tree in Matplotlib pixels.

        You can use the resolution parameter, the timings parameter, or neither of them, but not both. If you use
        neither, we will plot time points.
        """
        timings, use_hours = _get_timings(resolution, timings)
        if set_ylabel:
            axes.set_ylabel("Time (h)" if use_hours else "Time point")

        x_offset = x_offset_start
        for lineage in self.starting_tracks:
            if not lineage_filter(lineage):
                continue
            width = self._draw_single_lineage_colored(axes, lineage, x_offset, color_getter, label_getter, timings,
                                                      location_map, line_width)
            x_offset += width
        return x_offset - x_offset_start

    def _draw_single_lineage_colored(self, ax: Axes, lineage: LinkingTrack, x_offset: float, color_getter: _ColorGetter,
                                     label_getter: _LabelGetter, image_timings: ImageTimings,
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
                t0 = image_timings.get_time_h_since_start(time_point_min)
                label = label_getter(linking_track)
                if label is not None:
                    ax.text(x_offset + x + 0.05, t0 + 0.4, label, verticalalignment='top', clip_on=True)
                for time_point_of_line in range(time_point_min, time_point_max):
                    # get time points for current sub time interval i

                    t1 = image_timings.get_time_h_since_start(time_point_of_line + 1)
                    # get color corresponding to current z value
                    color_val_next = color_getter(time_point_of_line + 2, linking_track) if time_point_of_line + 2 <= time_point_max else None
                    if color_val_next != color_val:
                        # save line data
                        lines_XY.append([(x_offset + x, t0), (x_offset + x, t1)])
                        lines_col.append(color_val)

                        color_val = color_val_next
                        t0 = t1
                    location_map.set(x_offset + x, t1,
                                     linking_track.find_position_at_time_point_number(time_point_of_line + 1))
            if line.is_horizontal():
                linking_track = line.track

                # get indeces of timepoint prior to T
                time_point_of_line = line.time_point_number_start
                time = image_timings.get_time_h_since_start(time_point_of_line)

                # get color corresponding to current z value
                color_val_next = color_getter(time_point_of_line, linking_track)
                # save line data
                lines_XY.append(
                    [(x_offset + line.x_start, time), (x_offset + line.x_end, time)])
                lines_col.append(color_val_next)
                location_map.set_area(int(x_offset + line.x_start), int(time), int(x_offset + line.x_end), int(time),
                                      linking_track.find_position_at_time_point_number(time_point_of_line))

        line_segments = LineCollection(lines_XY, colors=lines_col, lw=line_width, capstyle="projecting")
        ax.add_collection(line_segments)

        return diagram_width

    def __repr__(self) -> str:
        return f"<Lineage tree of {len(self.starting_tracks)} cells>"
