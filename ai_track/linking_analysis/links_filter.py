"""Used to remove links that you don't want to analyze for some reason."""
from ai_track.core.experiment import Experiment


def delete_positions_without_links(experiment: Experiment):
    """Deletes all positions without any links."""
    positions = experiment.positions
    links = experiment.links

    # Find positions without links
    to_delete = list()
    for time_point in experiment.time_points():
        for position in positions.of_time_point(time_point):
            if len(links.find_links_of(position)) == 0:
                to_delete.append(position)

    # Actually delete them
    experiment.remove_positions(to_delete)


def delete_lineages_not_in_first_time_point(experiment: Experiment):
    """Used to delete lineages that did not reach the first time point."""
    first_time_point_number = experiment.positions.first_time_point_number()
    if first_time_point_number is None:
        return  # No data loaded

    # Find all positions to delete
    to_delete = list()
    for appeared_position in experiment.links.find_appeared_positions(first_time_point_number):
        appeared_track = experiment.links.get_track(appeared_position)
        for track in appeared_track.find_all_descending_tracks(include_self=True):
            for position in track.positions():
                to_delete.append(position)

    # Actually delete them
    experiment.remove_positions(to_delete)
