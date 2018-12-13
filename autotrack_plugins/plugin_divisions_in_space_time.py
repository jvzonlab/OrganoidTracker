"""On the x axis, we see the time. On the y axis, we see the distance along the crypt axis. For every cell, the length
of its cell cycle is recorded. On the graph, we see blocks representing the average cell cycle length in that space/time
area."""
from typing import List, Dict, Any

import numpy
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from autotrack.core import UserError, TimePoint
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking import cell_division_finder
from autotrack.linking_analysis import particle_age_finder

_TIME_POINTS_PER_CELL = 3
_PIXELS_PER_CELL = 5


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
        space_bucket = int(position_on_crypt_axis / _PIXELS_PER_CELL)
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
        "Graph//Cell cycle-Cell cycle lengths over space and time...": lambda: _show_graph(window)
    }


def _draw_graph(figure: Figure, grid: _SpaceTimeGrid):
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
    axes.set_title("Cell cycle lengths over space and time")


def _get_graphing_data(experiment: Experiment) -> _SpaceTimeGrid:
    grid = _SpaceTimeGrid()
    links = experiment.links
    if not links.has_links():
        raise UserError("No linking data found", "No links were loaded. Therefore, we cannot determine the cell age, so"
                                                 " we cannot plot anything.")

    # Rank all cells according to their crypt position, time point and cell cycle length
    families = cell_division_finder.find_families(links)
    i = 0
    for family in families:
        i+=1
        for particle in family.daughters:
            next_division = cell_division_finder.get_next_division(links, particle)
            if next_division is None:
                continue
            cell_cycle_length = particle_age_finder.get_age(links, next_division.mother)
            if cell_cycle_length is None:
                continue

            while True:
                next_particles = links.find_futures(particle)
                if len(next_particles) != 1:
                    break  # Found next division or end of cell track

                particle = next_particles.pop()
                time_point = particle.time_point()
                position_on_crypt_axis = experiment.data_axes.to_position_on_axis(particle)
                if position_on_crypt_axis is None:
                    continue

                grid.add_point(time_point, position_on_crypt_axis.pos, cell_cycle_length)
    return grid


def _show_graph(window: Window):
    grid = _get_graphing_data(window.get_experiment())

    if grid.is_empty():
        raise UserError("No cell cycles found", "No complete cell cycles were found in the linking data."
                                                " Cannot plot anything.")
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_graph(figure, grid))


# experiment = Experiment()
# links_extractor.add_data_to_experiment(experiment, r"S:\groups\zon-group\guizela\multiphoton\organoids\17-07-28_weekend_H2B-mCherry\nd799xy08-stacks\analyzed")
# cProfile.run("grid = _get_graphing_data(experiment)")
