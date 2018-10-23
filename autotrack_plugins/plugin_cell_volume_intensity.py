from typing import Optional, Tuple, List, Callable, Dict, Any

from matplotlib.figure import Figure
from networkx import Graph

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui import dialog, Window
from autotrack.linking import mother_finder
from statistics import median


GetStatistic = Callable[[Experiment, Particle], float]  # A function that gets some statistic of a cell, like its volume
PointList = Tuple[List[float], List[float]]  # A list of x values and a list of y values


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Cell cycle-Cell volumes over time...": lambda: _show_cell_volumes(window),
        "Graph/Cell cycle-Cell intensities over time...": lambda: _show_cell_intensities(window)
    }


def _show_cell_volumes(window: Window):
    def draw(figure: Figure):
        _plot_volumes(window.get_experiment(), figure)

    dialog.popup_figure(window.get_experiment().name, draw)


def _show_cell_intensities(window: Window):
    def draw(figure: Figure):
        _plot_intensities(window.get_experiment(), figure)

    dialog.popup_figure(window.get_experiment().name, draw)


def _get_volume(experiment: Experiment, particle: Particle) -> Optional[float]:
    shape = experiment.particles.get_shape(particle)
    try:
        return shape.volume()
    except NotImplementedError:
        return None


def _get_intensity(experiment: Experiment, particle: Particle) -> Optional[float]:
    shape = experiment.particles.get_shape(particle)
    try:
        return shape.intensity()
    except ValueError:
        return None


def _plot_volumes(experiment: Experiment, figure: Figure, mi_start=0, line_count=300, starting_time_point=50):
    """Plots the volumes of all cells in time. T=0 represents a cell division."""
    _plot_mother_stat(experiment, figure, _get_volume, 'Cell volume (px$^3$)', mi_start, line_count,
                      starting_time_point)


def _plot_mother_stat(experiment: Experiment, figure: Figure, stat: GetStatistic, y_label: str, mi_start: int,
                      line_count: int, starting_time_point: int):
    graph = experiment.links.get_baseline_else_scratch()
    if graph is None:
        raise UserError("No cell links", "No cell links were loaded, so we cannot track cell statistics over time.")
    mothers = [mother for mother in mother_finder.find_mothers(graph) if mother.time_point_number() >= starting_time_point]
    mothers = mothers[mi_start:mi_start + line_count]
    axes = figure.gca()
    show_legend = line_count <= 5

    all_values = []
    lines = []
    for mother in mothers:
        time_point_numbers, volumes = _data_into_past_until_division(experiment, mother, graph, stat)
        color = None if show_legend else (0, 0, 0, 0.2)
        lines.append(axes.plot(time_point_numbers, volumes, label=str(mother), color=color))
        all_values += volumes
    if len(all_values) == 0:
        raise UserError("No data to display", "No cell statistics were recorded. These data normally come from a "
                                              "Gaussian fit. Did you perform such a fit on the data?")
    axes.set_ylim(bottom=0, top=median(all_values) * 2)
    axes.set_xlabel('Time point')
    axes.set_ylabel(y_label)
    if show_legend:
        axes.legend()


def _plot_intensities(experiment: Experiment, figure: Figure, mi_start=0, line_count=300, starting_time_point=50):
    """Plots the intensities of all cells in time. T=0 represents a cell division."""
    _plot_mother_stat(experiment, figure, _get_intensity, 'Cell intensity (A.U.)', mi_start, line_count,
                      starting_time_point)


def _data_into_past_until_division(experiment: Experiment, starting_point: Particle, graph: Graph,
                                   func: GetStatistic) -> PointList:
    particle = starting_point
    x_values = []
    y_values = []
    while particle is not None:
        y_value = func(experiment, particle)
        if y_value is not None:
            x_values.append(particle.time_point_number() - starting_point.time_point_number())
            y_values.append(y_value)

        particle = _get_previous(particle, graph)
    return x_values, y_values


def _get_previous(particle: Particle, graph: Graph) -> Optional[Particle]:

    # Find the single previous position
    previous_positions = [p for p in graph[particle] if p.time_point_number() < particle.time_point_number()]
    if len(previous_positions) != 1:
        return None
    previous = previous_positions[0]

    # Find the single next position of the previous (ensures that we are not doing another cell division)
    next_positions = [p for p in graph[previous] if p.time_point_number() > previous.time_point_number()]
    if len(next_positions) != 1:
        return None  # This is a mother cell, so don't take it into account

    return previous

