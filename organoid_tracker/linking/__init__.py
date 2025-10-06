"""
Linking is the process of connecting cells from different time points, so that they can be followed over time.

We have two automatic linking functions in this package, as well as some supporting code.

The simplest linker is the nearest-neighbor linker:

>>> from organoid_tracker.linking import nearest_neighbor_linker
>>> from organoid_tracker.core.experiment import Experiment
>>> from organoid_tracker.core.position import Position
>>> experiment = Experiment()  # Placeholder, create an experiment with positions (but no links) here
>>> experiment.positions.add(Position(3, 0, 0, time_point_number=0))
>>> experiment.positions.add(Position(4, 0, 0, time_point_number=1))
>>>
>>> # Now lets do the linking
>>> experiment.links = nearest_neighbor_linker.nearest_neighbor(experiment)

The more complex linker is the dpct linker:
>>> from organoid_tracker.linking import dpct_linker
>>> all_potential_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
>>> # experiment must contain position data for 'division_penalty', 'appearance_penalty', 'dissappearance_penalty'
>>> # for all positions.
>>> # and must contain link_data for 'link_penalty' for all potential links
>>> experiment.links = dpct_linker.run(experiment.positions, all_potential_links,
>>>            link_weight=1, detection_weigh=1, division_weight=1, appearance_weight=1, dissappearance_weight=1)


"""
