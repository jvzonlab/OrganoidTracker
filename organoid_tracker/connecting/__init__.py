"""
This package is about establishing connections between cell :class:`~organoid_tracker.core.position.Position`s in the
same time point. You can use that to mark that cells are neighbors. (Although in principle you can connect any two
cells in the same time point.)

Note that connections are different from links, which are used to mark that two cells from subsequent time points
represent the same cell.

You can establish connections by hand in OrganoidTracker, or you can do it automatically. In this example,
we connect every cell to the 5 closest cells, with a maximum distance of 10 micrometer:

>>> from organoid_tracker.core.experiment import Experiment
>>> experiment = Experiment()  # Placeholder
>>> from organoid_tracker.connecting.connector_by_distance import ConnectorByDistance
>>> connector = ConnectorByDistance(max_distance_um=10, max_number=5)
>>> experiment.connections = connector.create_connections(experiment)

"""
