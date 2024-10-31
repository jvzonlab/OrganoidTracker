from typing import Iterable

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.typing import MPLColor

import numpy
import matplotlib.cm

from organoid_tracker.linking import cell_division_finder


def color_error_rates(time_point_number: int, track: LinkingTrack, experiment: Experiment) -> MPLColor:
    _COLORMAP = matplotlib.cm.get_cmap("RdYlGn")

    position = track.find_position_at_time_point_number(time_point_number)

    # create window around position
    prev_positions = _grab(experiment.links.iterate_to_past(position), start=1, stop=3)
    future_positions = _grab(experiment.links.iterate_to_future(position), start=1, stop=3)

    # iterate to find marginals
    marginals = []
    current_position = position

    for prev_position in prev_positions:
        marginal = (experiment.link_data.get_link_data(prev_position, current_position,
                                                       "marginal_probability"))
        if marginal is not None:
            marginals.append(marginal)

        current_position = prev_position

    current_position = position

    for next_position in future_positions:
        marginal = (experiment.link_data.get_link_data(current_position, next_position,
                                                       "marginal_probability"))
        if marginal is not None:
            marginals.append(marginal)

        current_position = next_position

    # take lowest prob in window to make errors visible
    if len(marginals) == 0:
        intensity = None
    else:
        intensity = numpy.log10(min(marginals) + 10 ** -10) - numpy.log10(1 - min(marginals) + 10 ** -10)

    if intensity is None:
        return 1, 1, 1

    return _COLORMAP((intensity) / 3.0)


def _grab(iterable: Iterable, start: int, stop: int) -> Iterable:
    """Grab a slice of an iterable. Like list(iterable)[min_index:max_index] but more efficient, as it doesn't need to
    unroll the entire iterable and store it in a list."""
    for i, item in enumerate(iterable):
        if i < start:
            continue
        if i >= stop:
            break
        yield item


def compute_lineage_error_probability(track: LinkingTrack, experiment: Experiment):
    # find all tracks downstream
    tracks = track.find_all_descending_tracks(include_self = True)

    probability = 1
    # multiply all probabilities except if an error is corrected
    for track in tracks:
        for pos in track.positions():
            prev_pos = experiment.links.find_single_past(pos)
            if prev_pos is not None:
                marginal_probability = experiment.link_data.get_link_data(prev_pos, pos, 'marginal_probability')
                error = experiment.position_data.get_position_data(pos, 'error') == 14
                suppressed_error = experiment.position_data.get_position_data(pos, 'suppressed_error') == 14
                if marginal_probability is not None:
                    if marginal_probability > 0.99 or (error and not suppressed_error):
                        probability = probability*marginal_probability

    return 1-probability


def compute_track_error_probability(track: LinkingTrack, experiment: Experiment):

    # multiply all probabilities except if an error is corrected
    probability = 1
    for pos in track.positions():

        prev_pos = experiment.links.find_single_past(pos)
        if prev_pos is not None:
            marginal_probability = experiment.link_data.get_link_data(prev_pos, pos, 'marginal_probability')
            error = experiment.position_data.get_position_data(pos, 'error') == 14
            suppressed_error = experiment.position_data.get_position_data(pos, 'suppressed_error') == 14
            if marginal_probability is not None:
                if marginal_probability > 0.99 or (error and not suppressed_error):
                    probability = probability*marginal_probability

    return 1-probability



