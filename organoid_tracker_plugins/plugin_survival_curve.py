from typing import List, Dict, Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.core import Color
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking_analysis import cell_fate_finder
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType
from organoid_tracker_plugins.plugin_cell_count_over_time import _plot

import numpy as np

def get_menu_items(window: Window):
    # This function is automatically called for any file named plugin_ ... .py in the plugins folder
    # You need to return a dictionary of menu options here
    return {
        "Graph//Cell cycle-Cell cycle//Divisions in survival curve...":
            lambda: _view_survival_curve(window)
    }


def _view_survival_curve(window: Window):
    minimum = dialog.prompt_int("Minimum cell cycle time (number of frames)", "what is the minimum plausible cell cycle length?",
                                minimum=0, maximum=100)

    lifetimes_all =[]
    sister_times_all = []
    divisions_all =[]
    sister_divisions_all = []
    names =[]

    for experiment in window.get_active_experiments():

        (lifetimes, sister_lifetimes,  sister_times, divisions, sister_divisions) = make_cell_cycle_table(experiment)
        # do not allow very short cell cycle times
        divisions[lifetimes<minimum] = 0
        sister_divisions[sister_lifetimes < minimum] = 0

        lifetimes_all.append(lifetimes)
        sister_times_all.append(sister_times)
        divisions_all.append(divisions)
        sister_divisions_all.append(sister_divisions)
        names.append(experiment.name)

    # cell cycle survival curve
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _plot_survival_curve(figure, lifetimes_all, divisions_all, names, 'Cell cycle time'))
    # sister pairs survival curve
    dialog.popup_figure(window.get_gui_experiment(),
                        lambda figure: _plot_survival_curve(figure, sister_times_all, sister_divisions_all, names, 'Time since sister division'))


def make_cell_cycle_table(experiment):
    # check for missed divisions in the tracks
    for link in experiment.links.find_all_links():
        division_penalty1 = experiment.position_data.get_position_data(link[0], 'division_penalty')
        division_penalty2 = experiment.position_data.get_position_data(link[1], 'division_penalty')

        next_position = experiment.links.find_single_future(link[1])
        if (next_position is not None) and (division_penalty1 is not None):
            division_penalty3 = experiment.position_data.get_position_data(next_position, 'division_penalty')

            # Is there a division detected for a window of time
            if (division_penalty1 + division_penalty2 + division_penalty3) / 3 < -2.0:
                track = experiment.links.get_track(link[0])
                if ((link[0].time_point_number() - track.min_time_point_number() > 6)
                        and (track.max_time_point_number() - link[0].time_point_number() > 6)):
                    experiment.links.remove_link(link[1], next_position)
            if (division_penalty1 is not None) and (division_penalty2 is not None) and (division_penalty3 is not None):
                if (division_penalty1 + division_penalty2 + division_penalty3) / 3 < -2.0:
                    track = experiment.links.get_track(link[0])
                    if ((link[0].time_point_number() - track.min_time_point_number() > 6)
                            and (track.max_time_point_number() - link[0].time_point_number() > 6)):
                        experiment.links.remove_link(link[1], next_position)

    lifetimes =[]
    divisions =[]
    sister_times =[]
    sister_lifetimes = []
    sister_divisions =[]

    for track in experiment.links.find_all_tracks():

        cell_id = experiment.links.get_track_id(track)
        first_pos = track.find_first_position()
        last_pos = track.find_last_position()

        start = first_pos.time_point_number()
        end = last_pos.time_point_number()

        division_penalty_start = experiment.position_data.get_position_data(first_pos, "division_penalty")
        division_penalty_end = experiment.position_data.get_position_data(last_pos, "division_penalty")

        if division_penalty_start is None:
            division_penalty_start = 1000
        if division_penalty_end is None:
            division_penalty_end = 1000

        # if a division is detected at the start of a track assign a start_division time
        if division_penalty_start < 0:
            start_division = start
        else:
            start_division = None

        # if the cell has a parent assign a start_division time
        parent_track = list(track.get_previous_tracks())
        if len(parent_track) > 0:
            start_division = start

        # does the track end in a division
        daughter_tracks = track.get_next_tracks()
        if (len(daughter_tracks) > 1) or (division_penalty_end < 0):
            end_division = end
        else:
            end_division = None

        # if the track starts with a division add it to the data
        if start_division is not None:
            divisions.append((end_division is not None))
            lifetimes.append(end - start)

        # add sister times
        if len(parent_track) > 0:

            # find sister
            daughter1, daughter2 = parent_track[0].get_next_tracks()
            sister_id = experiment.links.get_track_id(daughter1)
            if sister_id == cell_id:
                sister_id = experiment.links.get_track_id(daughter2)

            # does sister end in division
            sister_track = experiment.links.get_track_by_id(sister_id)
            sister_last_pos = sister_track.find_last_position()
            sister_division_penalty_end = experiment.position_data.get_position_data(sister_last_pos, "division_penalty")

            if sister_division_penalty_end is None:
                sister_division_penalty_end=1000

            cousin_tracks = sister_track.get_next_tracks()

            if ((len(cousin_tracks) > 1) or (sister_division_penalty_end < 0)) and len(list(sister_track.positions()))>30:
                sister_end_division = sister_last_pos.time_point_number()
            else:
                sister_end_division = None

            # if sister divides add to dataset
            if sister_end_division is not None:
                sister_divisions.append((end_division is not None))
                sister_lifetimes.append(end-start)
                sister_times.append(end - sister_end_division)

    return np.array(lifetimes), np.array(sister_lifetimes), np.array(sister_times), np.array(divisions), np.array(sister_divisions)


def _plot_survival_curve(figure, lifetimes_all, divisions_all, names, title):
    from lifelines import KaplanMeierFitter

    axes: Axes = figure.gca()
    kmf = KaplanMeierFitter()
    for lifetimes, divisions, name in zip(lifetimes_all, divisions_all, names):
        kmf.fit(lifetimes, divisions, label=name)
        kmf.plot_survival_function(ax=axes)

    axes.set_ylim(ymax=1.02, ymin=0)
    axes.set_xlabel('time (in frames)')
    axes.set_ylabel('non-divided fraction')

    axes.set_title(title)




