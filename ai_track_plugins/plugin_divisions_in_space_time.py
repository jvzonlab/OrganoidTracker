"""On the x axis, we see the time. On the y axis, we see the distance along the crypt axis. For every cell, the length
of its cell cycle is recorded. On the graph, we see blocks representing the average cell cycle length in that space/time
area."""
from typing import List, Dict, Any

import numpy
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ai_track.core import UserError, TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.linking import cell_division_finder
from ai_track.linking_analysis import particle_age_finder

_TIME_POINTS_PER_CELL = 3
_PIXELS_PER_CELL = 5
_OUT_OF_ORDINARY_PERCENTILE = 10


class _Cell:
    _total: int = 0
    _amount: int = 0

    def add_value(self, value: int):
        self._total += value
        self._amount += 1

    def average(self) -> float:
        if self._amount == 0:
            return 0
        return self._total / self._amount


class _SpaceTimeGrid:
    """Class to record the data of the graph"""

    _data: List[List[_Cell]]  # Outer lists are in time, inner lists in space

    def __init__(self):
        self._data = list()

    def add_point(self, time_point: TimePoint, position_on_crypt_axis: float, cell_cycle_length: int):
        """Records a single data point."""
        time_point_bucket = time_point.time_point_number() // _TIME_POINTS_PER_CELL
        while len(self._data) <= time_point_bucket:
            self._data.append([])  # Add empty lists

        space_list = self._data[time_point_bucket]
        space_bucket = max(0, int(position_on_crypt_axis / _PIXELS_PER_CELL))
        while len(space_list) <= space_bucket:
            space_list.append(_Cell())

        space_list[space_bucket].add_value(cell_cycle_length)

    def is_empty(self):
        """Returns True if add_point was never called."""
        return len(self._data) == 0

    def to_image(self):
        """Creates a 2D image of all the data points."""
        space_buckets = max([len(space_list) for space_list in self._data])
        time_buckets = len(self._data)
        image = numpy.zeros((space_buckets, time_buckets), dtype=numpy.float32)

        for time_bucket_index, space_bucket in enumerate(self._data):
            for space_bucket_index, cell in enumerate(space_bucket):
                image[space_bucket_index, time_bucket_index] = cell.average()

        return image


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell cycle//Space/time-Lengths over space and time//All cycles...": lambda: _show_cycle_lengths(window),
        "Graph//Cell cycle-Cell cycle//Space/time-Lengths over space and time//Longest...": lambda: _show_longest_cycles(window),
        "Graph//Cell cycle-Cell cycle//Space/time-Lengths over space and time//Shortest...": lambda: _show_shortest_cycles(window),
        "Graph//Cell cycle-Cell cycle//Space/time-Lengths over space and time//Middle...": lambda: _show_middle_cycles(window)
    }


def _draw_cycle_lengths(figure: Figure, grid: _SpaceTimeGrid, *, title: str = "Cell cycle lengths over space and time"):
    axes = figure.gca()
    image = grid.to_image()
    image_width = image.shape[1] * _TIME_POINTS_PER_CELL
    image_height = image.shape[0] * _PIXELS_PER_CELL

    image = axes.imshow(grid.to_image(), cmap="plasma", extent=(0, image_width, image_height, 0))
    divider = make_axes_locatable(axes)
    axes_on_right = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(image, cax=axes_on_right).set_label("Cell cycle length (time points)")

    axes.invert_yaxis()
    axes.set_xlabel("Time (time points)")
    axes.set_ylabel("Distance from crypt bottom (pixels)")
    axes.set_title(title)


def _get_graphing_data(experiment: Experiment, *, min_cycle_length: int = 0, max_cycle_length: int = 1000000
                       ) -> _SpaceTimeGrid:
    """Builds a grid of all cell cycle lengths in the experiment. Using min_cycle_length, you can exclude any cell
    cycle length that you think is too short."""
    grid = _SpaceTimeGrid()
    links = experiment.links
    if not links.has_links():
        raise UserError("No linking data found", "No links were loaded. Therefore, we cannot determine the cell age, so"
                                                 " we cannot plot anything.")

    # Rank all cells according to their crypt position, time point and cell cycle length
    families = cell_division_finder.find_families(links)
    i = 0
    for family in families:
        # Find all daughter cells that will divide again, record the duration of their upcoming cell cycle length
        i += 1
        for position in family.daughters:
            next_division = cell_division_finder.get_next_division(links, position)
            if next_division is None:
                continue
            cell_cycle_length = particle_age_finder.get_age(links, next_division.mother)
            if cell_cycle_length is None or cell_cycle_length < min_cycle_length \
                    or cell_cycle_length > max_cycle_length:
                continue

            while True:
                next_positions = links.find_futures(position)
                if len(next_positions) != 1:
                    break  # Found next division or end of cell track

                position = next_positions.pop()
                time_point = position.time_point()
                position_on_crypt_axis = experiment.splines.to_position_on_original_axis(experiment.links, position)
                if position_on_crypt_axis is None:
                    continue

                grid.add_point(time_point, position_on_crypt_axis.pos, cell_cycle_length)
    return grid


def _show_cycle_lengths(window: Window):
    grid = _get_graphing_data(window.get_experiment())
    if grid.is_empty():
        raise UserError("No cell cycles found", "No complete cell cycles were found in the linking data."
                                                " Cannot plot anything.")

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cycle_lengths(figure, grid))


def _show_longest_cycles(window: Window):
    experiment = window.get_experiment()

    all_cell_cycle_lengths = _get_all_cell_cycle_lengths(experiment.links)
    min_cycle_length = int(numpy.percentile(all_cell_cycle_lengths, 100 - _OUT_OF_ORDINARY_PERCENTILE))

    grid = _get_graphing_data(window.get_experiment(), min_cycle_length=min_cycle_length)
    if grid.is_empty():
        raise UserError("No cell cycles found", "No complete cell cycles were found in the linking data."
                                                " Cannot plot anything.")

    title = f"Cell cycle lengths over space and time of {_OUT_OF_ORDINARY_PERCENTILE}% longest cycles"
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cycle_lengths(figure, grid, title=title))


def _show_shortest_cycles(window: Window):
    experiment = window.get_experiment()

    all_cell_cycle_lengths = _get_all_cell_cycle_lengths(experiment.links)
    max_cycle_length = int(numpy.percentile(all_cell_cycle_lengths, _OUT_OF_ORDINARY_PERCENTILE))

    grid = _get_graphing_data(window.get_experiment(), max_cycle_length=max_cycle_length)
    if grid.is_empty():
        raise UserError("No cell cycles found", "No complete cell cycles were found in the linking data."
                                                " Cannot plot anything.")

    title = f"Cell cycle lengths over space and time of {_OUT_OF_ORDINARY_PERCENTILE}% shortest cycles"
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cycle_lengths(figure, grid, title=title))


def _show_middle_cycles(window: Window):
    experiment = window.get_experiment()

    all_cell_cycle_lengths = _get_all_cell_cycle_lengths(experiment.links)
    min_cycle_length = int(numpy.percentile(all_cell_cycle_lengths, 50 - _OUT_OF_ORDINARY_PERCENTILE/2))
    max_cycle_length = int(numpy.percentile(all_cell_cycle_lengths, 50 + _OUT_OF_ORDINARY_PERCENTILE/2))

    grid = _get_graphing_data(window.get_experiment(), min_cycle_length=min_cycle_length, max_cycle_length=max_cycle_length)
    if grid.is_empty():
        raise UserError("No cell cycles found", "No complete cell cycles were found in the linking data."
                                                " Cannot plot anything.")

    title = f"Cell cycle lengths over space and time of {_OUT_OF_ORDINARY_PERCENTILE}% middle-length cycles"
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cycle_lengths(figure, grid, title=title))


def _get_all_cell_cycle_lengths(links: Links) -> List[int]:
    """Gets all cell cycle lengths that are in the experiment."""
    all_cell_cycle_lengths = []
    for family in cell_division_finder.find_families(links):
        for daughter in family.daughters:
            next_division = cell_division_finder.get_next_division(links, daughter)
            if next_division is None:
                continue
            cell_cycle_length = particle_age_finder.get_age(links, next_division.mother)
            if cell_cycle_length is None:
                continue
            all_cell_cycle_lengths.append(cell_cycle_length)
    return all_cell_cycle_lengths
