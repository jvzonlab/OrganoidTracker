from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking.nearby_position_finder import find_closest_n_positions
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker, StartMarker
import numpy as np


def postprocess(experiment: Experiment, margin_xy: int):
    _remove_positions_close_to_edge(experiment, margin_xy)
    # _remove_spurs(experiment)
    _mark_positions_going_out_of_image(experiment)


def finetune_solution(experiment: Experiment, experiment_result: Experiment):
    """Adds, deletes or swaps single links to lower the energy of the solution"""
    mothers = cell_division_finder.find_mothers(experiment_result.links, exclude_multipolar=False)

    # removes links that are best replaced by appearances + disappearances
    for position in experiment_result.positions:

        prev_positions = list(experiment_result.links.find_pasts(position))

        if len(prev_positions) == 1:
            prev_position = prev_positions[0]

            if prev_position in mothers:
                old_penalty = experiment_result.link_data.get_link_data(prev_position, position, 'link_penalty') \
                              + experiment_result.positions.get_position_data(prev_position, 'division_penalty')
                new_penalty = experiment_result.positions.get_position_data(position, 'appearance_penalty')
            else:
                old_penalty = experiment_result.link_data.get_link_data(prev_position, position, 'link_penalty')
                new_penalty = experiment_result.positions.get_position_data(position, 'appearance_penalty') \
                              + experiment_result.positions.get_position_data(prev_position,
                                                                                  'disappearance_penalty')

            if old_penalty > new_penalty:
                experiment_result.links.remove_link(prev_position, position)

    # connect loose starts by breaking/appending other tracks
    loose_starts = list(experiment_result.links.find_appeared_positions(
        time_point_number_to_ignore=experiment.first_time_point_number()))

    for position in loose_starts:
        old_appearance_penalty = experiment_result.positions.get_position_data(position, 'appearance_penalty')

        prev_positions = experiment.links.find_pasts(position)

        source_position = None
        old_target_position = None
        min_penalty_diff = 0

        for prev_position in prev_positions:
            new_link_penalty = experiment.link_data.get_link_data(prev_position, position, 'link_penalty')

            next_positions = experiment_result.links.find_futures(prev_position)

            for next_position in next_positions:
                old_link_penalty = experiment.link_data.get_link_data(prev_position, next_position, 'link_penalty')
                new_appearance_penalty = experiment_result.positions.get_position_data(next_position,
                                                                                           'appearance_penalty')
                penalty_diff = (-old_link_penalty - old_appearance_penalty + new_link_penalty + new_appearance_penalty)

                if penalty_diff < min_penalty_diff:
                    source_position = prev_position
                    old_target_position = next_position
                    min_penalty_diff = penalty_diff

            if len(next_positions) == 0:
                old_disappearance_penalty = experiment_result.positions.get_position_data(prev_position,
                                                                                              'disappearance_penalty')
                penalty_diff = (-old_disappearance_penalty - old_appearance_penalty + new_link_penalty)

                if penalty_diff < min_penalty_diff:
                    source_position = prev_position
                    old_target_position = None
                    min_penalty_diff = penalty_diff

        if source_position is not None:
            if old_target_position is not None:
                experiment_result.links.remove_link(source_position, old_target_position)
                experiment_result.links.add_link(source_position, position)
            else:
                experiment_result.links.add_link(source_position, position)

    # connect loose ends by breaking/preceding other tracks
    loose_ends = list(experiment_result.links.find_disappeared_positions(
        time_point_number_to_ignore=experiment.last_time_point_number()))

    for position in loose_ends:
        old_disappearance_penalty = experiment_result.positions.get_position_data(position, 'disappearance_penalty')

        next_positions = experiment.links.find_futures(position)

        target_position = None
        old_source_position = None
        min_penalty_diff = 0

        for next_position in next_positions:
            new_link_penalty = experiment.link_data.get_link_data(position, next_position, 'link_penalty')

            prev_positions = experiment_result.links.find_pasts(next_position)

            for prev_position in prev_positions:
                old_link_penalty = experiment.link_data.get_link_data(prev_position, next_position, 'link_penalty')
                new_disappearance_penalty = experiment_result.positions.get_position_data(prev_position,
                                                                                              'disappearance_penalty')
                penalty_diff = (
                            -old_link_penalty - old_disappearance_penalty + new_link_penalty + new_disappearance_penalty)

                if penalty_diff < min_penalty_diff:
                    target_position = next_position
                    old_source_position = prev_position
                    min_penalty_diff = penalty_diff

            if len(prev_positions) == 0:
                old_appearance_penalty = experiment_result.positions.get_position_data(next_position,
                                                                                           'appearance_penalty')

                penalty_diff = (-old_appearance_penalty - old_disappearance_penalty + new_link_penalty)

                if penalty_diff < min_penalty_diff:
                    target_position = next_position
                    min_penalty_diff = penalty_diff

        if target_position is not None:
            if old_source_position is not None:
                experiment_result.links.remove_link(old_source_position, target_position)
                experiment_result.links.add_link(position, target_position)
            else:
                experiment_result.links.add_link(position, target_position)

    # add links to possible divisions
    for position in experiment_result.positions:
        next_positions = list(experiment_result.links.find_futures(position))
        next_possible_positions = list(experiment.links.find_futures(position))

        division_penalty = experiment.positions.get_position_data(position, 'division_penalty')

        # check if cell is not currently dividing
        if len(next_positions) == 1:

            for next_possible_position in next_possible_positions:

                prev_position = list(experiment_result.links.find_pasts(next_possible_position))

                if (next_possible_position not in next_positions) and (len(prev_position) == 1):
                    prev_position = prev_position[0]

                    old_link_penalty = experiment.link_data.get_link_data(prev_position, next_possible_position,
                                                                          'link_penalty')
                    new_link_penalty = experiment.link_data.get_link_data(position, next_possible_position,
                                                                          'link_penalty')
                    new_disappearance_penalty = experiment_result.positions.get_position_data(prev_position,
                                                                                                  'disappearance_penalty')

                    if division_penalty + new_link_penalty + new_disappearance_penalty < old_link_penalty:
                        experiment_result.links.remove_link(prev_position, next_possible_position)
                        experiment_result.links.add_link(position, next_possible_position)
                        break

                elif (next_possible_position not in next_positions) and (len(prev_position) == 0):
                    new_link_penalty = experiment.link_data.get_link_data(position, next_possible_position,
                                                                          'link_penalty')
                    old_appearance_penalty = experiment_result.positions.get_position_data(next_possible_position,
                                                                                               'appearance_penalty')

                    if division_penalty + new_link_penalty < old_appearance_penalty:
                        experiment_result.links.add_link(position, next_possible_position)
                        break

    # swap links around
    for position in experiment_result.positions:
        unfixed = True
        next_positions = list(experiment_result.links.find_futures(position))

        for next_position in next_positions:

            past_positions = list(experiment.links.find_pasts(next_position))

            if len(past_positions) > 1:
                past_positions.remove(position)

                for past_position in past_positions:

                    alternative_next_positions = list(experiment_result.links.find_futures(past_position))

                    for alternative_next_position in alternative_next_positions:

                        old_link_penalty = experiment.link_data.get_link_data(position, next_position,
                                                                              'link_penalty')
                        old_link_penalty2 = experiment.link_data.get_link_data(past_position, alternative_next_position,
                                                                               'link_penalty')
                        new_link_penalty = experiment.link_data.get_link_data(past_position, next_position,
                                                                              'link_penalty')

                        if experiment.links.contains_link(position, alternative_next_position):
                            new_link_penalty2 = experiment.link_data.get_link_data(position, alternative_next_position,
                                                                                   'link_penalty')
                            break_track = False
                        # if we do not swap two links, but change a link anc create a disappearance + an appearance
                        else:
                            new_link_penalty2 = \
                                experiment.positions.get_position_data(position, 'disappearance_penalty') \
                                + experiment.positions.get_position_data(alternative_next_position,
                                                                             'appearance_penalty')
                            break_track = True

                        if unfixed and (old_link_penalty + old_link_penalty2 > new_link_penalty + new_link_penalty2):
                            unfixed = False

                            if break_track == False:
                                experiment_result.links.add_link(position, alternative_next_position)

                            experiment_result.links.add_link(past_position, next_position)

                            experiment_result.links.remove_link(past_position, alternative_next_position)
                            experiment_result.links.remove_link(position, next_position)

    # remove unlikely divisions
    mothers = cell_division_finder.find_mothers(experiment_result.links, exclude_multipolar=False)

    for position in mothers:
        if experiment_result.positions.get_position_data(position, 'division_penalty') > 2.0:

            next_positions = list(experiment_result.links.find_futures(position))

            if experiment.link_data.get_link_data(position, next_positions[0],
                                                  'link_penalty') > experiment.link_data.get_link_data(position,
                                                                                                       next_positions[
                                                                                                           1],
                                                                                                       'link_penalty'):
                experiment_result.links.remove_link(position, next_positions[0])
            else:
                experiment_result.links.remove_link(position, next_positions[1])

    return experiment_result


def connect_loose_ends(experiment: Experiment, experiment_result: Experiment, oversegmentation_penalty=2.0, window=4):
    """connects tracks broken up by overgsegmentation (---===---- -> ---------)"""

    # find loose starts
    loose_starts = list(experiment_result.links.find_appeared_positions(
        time_point_number_to_ignore=experiment.first_time_point_number()))
    starts_plus_window = loose_starts

    for position in loose_starts:
        future_positions = list(experiment_result.links.iterate_to_future(position))
        starts_plus_window = starts_plus_window + future_positions[0:min(len(future_positions), window)]

    # order the loose ends in time to avoid mix-ups
    starts_plus_window_ordered = []

    for time_point in experiment.time_points():
        for position in experiment.positions.of_time_point(time_point):
            if position in starts_plus_window:
                starts_plus_window_ordered.append(position)

    # find loose ends
    loose_ends = list(experiment_result.links.find_disappeared_positions(
        time_point_number_to_ignore=experiment.last_time_point_number()))
    ends_plus_window = loose_ends

    for position in loose_ends:
        track = experiment_result.links.get_track(position)
        time_point_number = position.time_point_number()
        past_positions = []

        for t in range(0, window):
            if (time_point_number - t) > track.first_time_point_number():
                past_positions.append(track.find_position_at_time_point_number(time_point_number - t))

        ends_plus_window = ends_plus_window + past_positions

    # remember which tracks are already fixed
    fixed_ends = []
    fixed_starts = []
    oversegmentations_fixed = []

    # cycle over all loose starts
    for position in starts_plus_window_ordered:

        if experiment_result.links.get_track(position) is not None:
            positions_track = experiment_result.links.get_track(position).positions()

            # all possible connections
            past_positions = experiment.links.find_pasts(position)

            for past_position in past_positions:

                # is the past position eligble?
                if ((past_position in ends_plus_window) and (past_position not in positions_track) and
                        (past_position not in fixed_ends) and (position not in fixed_starts) and
                        (experiment_result.links.get_track(position) is not experiment_result.links.get_track(
                            past_position))):

                    link_penalty = experiment.link_data.get_link_data(past_position, position,
                                                                      'link_penalty')
                    disappearance_penalty = experiment.positions.get_position_data(past_position,
                                                                                       'disappearance_penalty')
                    appearance_penalty = experiment.positions.get_position_data(position,
                                                                                    'appearance_penalty')

                    # connect tracks and remove spurious positions
                    if link_penalty + oversegmentation_penalty < disappearance_penalty + appearance_penalty:
                        # remove oversegmentations
                        remove_past_positions = list(experiment_result.links.iterate_to_past(position))
                        remove_past_positions.pop(0)

                        remove_future_positions = list(experiment_result.links.iterate_to_future(past_position))
                        remove_future_positions.pop(0)

                        experiment_result.remove_positions(remove_past_positions + remove_future_positions)

                        experiment.remove_positions(
                            remove_past_positions + remove_future_positions)  # Also change set of positions

                        # add connecting link
                        experiment_result.links.add_link(past_position, position)

                        # remember which tracks are now fixed
                        future_positions = list(experiment_result.links.iterate_to_future(position))
                        future_positions = future_positions[:min(len(future_positions), window + 1)]

                        past_positions = list(experiment_result.links.iterate_to_past(position))
                        past_positions = past_positions[:min(len(past_positions), window + 1)]

                        fixed_starts = fixed_starts + remove_past_positions + remove_future_positions + future_positions
                        fixed_ends = fixed_ends + remove_past_positions + remove_future_positions + past_positions

                        # counter
                        oversegmentations_fixed = oversegmentations_fixed + remove_past_positions + remove_future_positions

                        # update link data to include oversegmentation penalty
                        experiment.link_data.set_link_data(past_position, position, 'link_penalty',
                                                           link_penalty + oversegmentation_penalty)
                        experiment.link_data.set_link_data(past_position, position, 'link_probability',
                                                           10 ** -(link_penalty + oversegmentation_penalty) / (10 ** -(
                                                                       link_penalty + oversegmentation_penalty) + 1))

                        experiment_result.link_data.set_link_data(past_position, position, 'link_penalty',
                                                                  link_penalty + oversegmentation_penalty)
                        experiment_result.link_data.set_link_data(past_position, position, 'link_probability',
                                                                  10 ** -(link_penalty + oversegmentation_penalty) / (
                                                                              10 ** -(
                                                                                  link_penalty + oversegmentation_penalty) + 1))

    print('number of oversegmentations fixed:')
    print(len(oversegmentations_fixed))

    return experiment_result, experiment


def bridge_gaps(experiment: Experiment, experiment_result: Experiment, miss_penalty=2.0):
    """connects tracks broken up by missed cell division (----x---- -> ---------)"""

    # find loose starts and ends
    loose_starts = list(experiment_result.links.find_appeared_positions(
        time_point_number_to_ignore=experiment.first_time_point_number()))

    loose_ends = list(experiment_result.links.find_disappeared_positions(
        time_point_number_to_ignore=experiment.last_time_point_number()))

    # if a position can be division but is not now, it can be considered a loose end as well
    for position in experiment_result.positions:
        if (experiment_result.positions.get_position_data(position, 'division_penalty') < 0) and (
                len(experiment_result.links.find_futures(position)) == 1):
            loose_ends.append(position)

    # remember which gaps are already fixed
    fixed = []

    for position in loose_ends:

        # find 6 closest neighbors in the the frame after the potential gap
        prev_time_point = position.time_point()
        next_time_point = TimePoint(position.time_point_number() + 2)
        neighbors = list(find_closest_n_positions(experiment_result.positions.of_time_point(next_time_point),
                                                  around=position, max_amount=6, max_distance_um=10,
                                                  resolution=experiment.images.resolution()))
        neighbors.reverse()

        # always include closest neighbor (adds a scale-free element to it)
        # Probably better to replace this completely by a more adapative distance threshhold
        if len(neighbors)==0:
            neighbors = list(find_closest_n_positions(experiment_result.positions.of_time_point(next_time_point),
                                                  around=position, max_amount=1,
                                                  resolution=experiment.images.resolution()))

        for neighbor in neighbors:
            distance = position.distance_um(neighbor, resolution=experiment.images.resolution())

            if (neighbor in loose_starts):
                # if we find a candidate to link with we also want to make sure that there is no better option for this candidate to link up to
                alternative_ends = find_closest_n_positions(experiment_result.positions.of_time_point(prev_time_point),
                                                            around=neighbor, max_amount=6, max_distance_um=10,
                                                            resolution=experiment.images.resolution())

                closest_alternative = position
                closest_distance = distance

                for alternative_end in alternative_ends:

                    if ((alternative_end.distance_um(neighbor,
                                                     resolution=experiment.images.resolution()) < closest_distance)
                            and (alternative_end in loose_ends)):
                        closest_distance = alternative_end.distance_um(neighbor,
                                                                       resolution=experiment.images.resolution())
                        closest_alternative = alternative_end

                # if the candidate is the best option then we can connect the tracks
                if (position == closest_alternative) and (position not in fixed) and (neighbor not in fixed):
                    disappearance_penalty = experiment.positions.get_position_data(position,
                                                                                       'disappearance_penalty')
                    appearance_penalty = experiment.positions.get_position_data(neighbor,
                                                                                    'appearance_penalty')

                    if miss_penalty < disappearance_penalty + appearance_penalty:

                        fixed.append(position)
                        fixed.append(neighbor)

                        # create position
                        time_point = TimePoint(position.time_point_number() + 1)

                        add_position = Position(x=round(0.5 * neighbor.x + 0.5 * position.x),
                                                y=round(0.5 * neighbor.y + 0.5 * position.y),
                                                z=round(0.5 * neighbor.z + 0.5 * position.z),
                                                time_point=time_point)

                        experiment_result.positions.add(add_position)
                        experiment_result.positions.set_position_data(add_position, 'division_penalty', value=10.0)

                        # add links
                        experiment_result.links.add_link(position, add_position)
                        experiment_result.links.add_link(add_position, neighbor)

                        # create position and links in full graph
                        experiment.positions.add(add_position)
                        experiment.positions.set_position_data(add_position, 'division_penalty', value=10.0)
                        experiment.positions.set_position_data(add_position, 'division_probability', value=0)

                        # using the position (i.e. not letting it disappear) is associated with a penalty
                        experiment.positions.set_position_data(add_position, 'disappearance_penalty',
                                                                   value=-miss_penalty)  # +appearance_penalty)
                        experiment.positions.set_position_data(add_position, 'appearance_penalty',
                                                                   value=-miss_penalty)  # +appearance_penalty)

                        # add all possible links for later marginalization withg uniform probabilities
                        alternatives = list(
                            find_closest_n_positions(experiment_result.positions.of_time_point(prev_time_point),
                                                     around=position, max_amount=6, max_distance_um=7,
                                                     resolution=experiment.images.resolution())) \
                                       + list(
                            find_closest_n_positions(experiment_result.positions.of_time_point(next_time_point),
                                                     around=neighbor, max_amount=6, max_distance_um=7,
                                                     resolution=experiment.images.resolution()))

                        link_probability = 0.5  # 1/((len(alternatives)+2)/2 + 1)
                        link_penalty = np.log10((1 - link_probability + 10 ** -10) / (link_probability + 10 ** -10))

                        # link
                        experiment.links.add_link(position, add_position)
                        experiment.links.add_link(add_position, neighbor)
                        experiment.link_data.set_link_data(position, add_position, 'link_penalty', value=link_penalty)
                        experiment.link_data.set_link_data(add_position, neighbor, 'link_penalty', value=link_penalty)

                        for alternative in alternatives:
                            experiment.links.add_link(add_position, alternative)
                            experiment.link_data.set_link_data(add_position, alternative, 'link_penalty',
                                                               value=link_penalty)

    print('number of gaps fixed:')
    print(len(fixed) // 2)

    return experiment_result, experiment


def bridge_gaps2(experiment: Experiment, experiment_result: Experiment, miss_penalty=2.0):
    """connects tracks broken up by not having a proposed link between them (----____ -> ---------)"""
    # find loose starts and ends
    loose_starts = list(experiment_result.links.find_appeared_positions(
        time_point_number_to_ignore=experiment.first_time_point_number()))

    loose_ends = list(experiment_result.links.find_disappeared_positions(
        time_point_number_to_ignore=experiment.last_time_point_number()))

    # if a position can be division but is not now, it can be considered a loose end as well
    for position in experiment_result.positions:
        if (experiment_result.positions.get_position_data(position, 'division_penalty') < 0) and (
                len(experiment_result.links.find_futures(position)) == 1):
            loose_ends.append(position)

    # remember which gaps are already fixed
    fixed = []

    for position in loose_ends:

        # find 6 closest neighbors in the current frame
        time_point = position.time_point()
        neighbors = list(find_closest_n_positions(experiment_result.positions.of_time_point(time_point),
                                                  around=position, max_amount=6, max_distance_um=7,
                                                  resolution=experiment.images.resolution()))
        neighbors.reverse()

        for neighbor in neighbors:
            distance = position.distance_um(neighbor, resolution=experiment.images.resolution())

            if (neighbor in loose_starts):
                # if we find a candidate to link with we also want to make sure that there is no better option
                alternative_ends = find_closest_n_positions(experiment_result.positions.of_time_point(time_point),
                                                            around=neighbor, max_amount=6, max_distance_um=7,
                                                            resolution=experiment.images.resolution())

                closest_alternative = position
                closest_distance = distance

                for alternative_end in alternative_ends:

                    if ((alternative_end.distance_um(neighbor,
                                                     resolution=experiment.images.resolution()) < closest_distance)
                            and (alternative_end in loose_ends)):
                        closest_distance = alternative_end.distance_um(neighbor,
                                                                       resolution=experiment.images.resolution())
                        closest_alternative = alternative_end

                # if the candidate is the best option then we can connect the tracks
                if (position == closest_alternative) and (position not in fixed) and (neighbor not in fixed):
                    disappearance_penalty = experiment.positions.get_position_data(position,
                                                                                       'disappearance_penalty')
                    appearance_penalty = experiment.positions.get_position_data(neighbor,
                                                                                    'appearance_penalty')

                    if miss_penalty < disappearance_penalty + appearance_penalty:
                        fixed.append(position)
                        fixed.append(neighbor)

                        prev_position = experiment_result.links.find_single_past(position)

                        # sometimes no previous position is found because it was removed during a previous gap closing operation
                        # (this can happen with tracks that are only two frames long)
                        if prev_position is None:
                            continue

                        # distinguish between loose ends and potential divisions
                        if len(experiment_result.links.find_futures(position)) == 0:
                            experiment_result.remove_position(position)
                            experiment.remove_position(position)

                        # using the link is associated with a penalty
                        experiment_result.links.add_link(prev_position, neighbor)
                        experiment.links.add_link(prev_position, neighbor)
                        experiment.link_data.set_link_data(prev_position, neighbor, 'link_penalty', value=miss_penalty)

    print('number of gaps fixed:')
    print(len(fixed) // 2)

    return experiment_result, experiment


def pinpoint_divisions(experiment: Experiment, experiment_result: Experiment, min_penalty_diff=1.0):
    """if two cell detections are made before the division network expects a division we want to align these events """

    mothers = cell_division_finder.find_mothers(experiment_result.links, exclude_multipolar=True)
    for position in mothers:

        next_positions = list(experiment_result.links.find_futures(position))

        # how do the division scores of mothers and daughters compare
        div_penalty = experiment_result.positions.get_position_data(position, 'division_penalty')
        div_penalty_0 = experiment_result.positions.get_position_data(next_positions[0], 'division_penalty')
        div_penalty_1 = experiment_result.positions.get_position_data(next_positions[1], 'division_penalty')

        # needed?
        if div_penalty is None:
            div_penalty = 2.0
        if div_penalty_0 is None:
            div_penalty_0 = 2.0
        if div_penalty_1 is None:
            div_penalty_1 = 2.0

        # move the division event to position_0
        if (div_penalty_0 < div_penalty_1) and (div_penalty_0 + min_penalty_diff < div_penalty) and (
                next_positions[0] not in mothers):

            next_next_positions = list(experiment_result.links.find_futures(next_positions[1]))
            if (len(next_next_positions) == 1):
                next_next_position = next_next_positions[0]

                if experiment.links.contains_link(next_positions[0], next_next_position):
                    experiment_result.remove_position(next_positions[1])
                    experiment.remove_position(next_positions[1])

                    experiment_result.links.add_link(next_positions[0], next_next_position)

        # move the division event to position_1
        elif (div_penalty_1 + min_penalty_diff < div_penalty) and (next_positions[1] not in mothers):
            next_next_positions = list(experiment_result.links.find_futures(next_positions[0]))
            if (len(next_next_positions) == 1):
                next_next_position = next_next_positions[0]

                if experiment.links.contains_link(next_positions[1], next_next_position):
                    experiment_result.remove_position(next_positions[0])
                    experiment.remove_position(next_positions[0])

                    experiment_result.links.add_link(next_positions[1], next_next_position)

    return experiment_result, experiment


def remove_tracks_too_deep(experiment: Experiment, max_z: int):
    loose_ends = list(experiment.links.find_disappeared_positions())

    for position in loose_ends:

        offset = experiment.images.offsets.of_time_point(position.time_point())

        if position.z > (max_z + offset.z):

            # check if the start of the track was also too deep
            pasts = list(experiment.links.iterate_to_past(position))

            if len(pasts) > 0:
                first_position = pasts[-1]

                offset = experiment.images.offsets.of_time_point(first_position.time_point())

                if first_position.z > (max_z + offset.z):
                    # remove track
                    experiment.remove_positions(pasts)
                    experiment.remove_position(position)

    return experiment


def _remove_tracks_too_short(experiment_result: Experiment, experiment: Experiment, min_t: int):
    loose_start = list(
        experiment_result.links.find_appeared_positions(
            time_point_number_to_ignore=experiment.first_time_point_number()))

    for position in loose_start:
        track = experiment_result.links.get_track(position)
        if track is not None:
            to_remove = list(track.positions())

            next_tracks = track.find_all_descending_tracks()

            for next_track in next_tracks:
                to_remove = to_remove + list(next_track.positions())

            if (len(to_remove) <= min_t) and (track.last_time_point_number()!=experiment.last_time_point_number()):
                experiment_result.remove_positions(to_remove)
                experiment.remove_positions(to_remove)

    return experiment_result, experiment


def _remove_spurs_division(experiment: Experiment, experiment_result: Experiment):
    mothers = cell_division_finder.find_mothers(experiment_result.links, exclude_multipolar=True)

    for position in mothers:
        daughters = list(experiment_result.links.find_futures(position))
        division_penalty = experiment.positions.get_position_data(position, 'division_penalty')

        for daughter in daughters:

            if (len(experiment_result.links.find_futures(daughter)) == 0) and (division_penalty > 0):
                experiment_result.remove_position(daughter)
                experiment.remove_position(daughter)

    return experiment_result, experiment


def _remove_positions_close_to_edge(experiment: Experiment, margin_xy: int):
    image_loader = experiment.images
    links = experiment.links
    positions = experiment.positions
    for time_point in experiment.time_points():
        for position in list(experiment.positions.of_time_point(time_point)):
            if not image_loader.is_inside_image(position, margin_xy=margin_xy):
                # Remove cell, but inform neighbors first
                _add_out_of_view_markers(links, positions, position)
                experiment.remove_position(position, update_splines=False)


def _mark_positions_going_out_of_image(experiment: Experiment):
    """Adds "going into view" and "going out of view" markers to all positions that fall outside the next or previous
    image, in case the camera was moved."""
    for time_point in experiment.time_points():
        try:
            time_point_previous = experiment.get_previous_time_point(time_point)
        except ValueError:
            continue  # This is the first time point

        offset = experiment.images.offsets.of_time_point(time_point)
        offset_previous = experiment.images.offsets.of_time_point(time_point_previous)
        if offset == offset_previous:
            continue  # Image didn't move, so no positions can go out of the view

        for position in experiment.positions.of_time_point(time_point_previous):
            # Check for positions in the previous image that fall outside the current image
            if not experiment.images.is_inside_image(position.with_time_point(time_point)):
                linking_markers.set_track_end_marker(experiment.positions, position, EndMarker.OUT_OF_VIEW)

        for position in experiment.positions.of_time_point(time_point):
            # Check for positions in the current image that fall outside the previous image
            if not experiment.images.is_inside_image(position.with_time_point(time_point_previous)):
                linking_markers.set_track_start_marker(experiment.positions, position, StartMarker.GOES_INTO_VIEW)


def _add_out_of_view_markers(links: Links, positions: PositionCollection, position: Position):
    """Adds markers to the remaining links so that it is clear why they appeared/disappeared."""
    linked_positions = links.find_links_of(position)
    for linked_position in linked_positions:
        if linked_position.time_point_number() < position.time_point_number():
            linking_markers.set_track_end_marker(positions, linked_position, EndMarker.OUT_OF_VIEW)
        else:
            linking_markers.set_track_start_marker(positions, linked_position, StartMarker.GOES_INTO_VIEW)


def _remove_spurs(experiment: Experiment):
    """Removes all very short tracks that end in a cell death."""
    links = experiment.links
    for position in list(links.find_appeared_positions()):
        _check_for_and_remove_spur(experiment, links, position)


def _remove_single_positions(experiment_result: Experiment, experiment: Experiment):
    """Removes loose positions."""
    positions = experiment_result.positions
    to_remove = []

    for position in positions:
        if (len(experiment_result.links.find_futures(position)) +
            len(experiment_result.links.find_pasts(position))) == 0:
            to_remove.append(position)

    experiment.remove_positions(to_remove)
    experiment_result.remove_positions(to_remove)

    return experiment_result, experiment


def _check_for_and_remove_spur(experiment: Experiment, links: Links, position: Position):
    track_length = 0
    positions_in_track = [position]

    while True:
        next_positions = links.find_futures(position)
        if len(next_positions) == 0:
            # End of track
            if track_length < 3:
                # Remove this track, it is too short
                for position_in_track in positions_in_track:
                    experiment.remove_position(position_in_track, update_splines=False)
            return
        if len(next_positions) > 1:
            # Cell division
            for next_position in next_positions:
                _check_for_and_remove_spur(experiment, links, next_position)
            return

        position = next_positions.pop()
        positions_in_track.append(position)
        track_length += 1
