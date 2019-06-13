from typing import Dict, Any, List

from matplotlib.figure import Figure

from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers, particle_age_finder


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//Graph-Shedding events following divisions...": lambda: _show_sheddings_following_divisions(window)
    }





def _show_sheddings_following_divisions(window: Window):
    experiment = window.get_experiment()

    shed_ages_hours = _get_shed_durations_after_division_hours(experiment)
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure, shed_ages_hours), size_cm=(8, 6))


def _get_shed_durations_after_division_hours(experiment: Experiment) -> List[float]:
    shed_ages_hours = []
    time_point_interval_h = experiment.images.resolution().time_point_interval_h

    for shed_position in linking_markers.find_shed_positions(experiment.links):
        age = particle_age_finder.get_age(experiment.links, shed_position)
        if age is None:
            continue
        shed_ages_hours.append(age * time_point_interval_h)

    return shed_ages_hours


def _draw_figure(figure: Figure, shed_ages_hours: List[float]):
    axes = figure.gca()
    axes.hist(shed_ages_hours)
    axes.set_xlabel("Time since their last division (h)")
    axes.set_ylabel("Amount of shed cells")
    figure.tight_layout()
