"""
The code used to laod and save files in Guizela's old tracking format. Necessary so that we can still use the tracking
data that predates OrganoidTracker.

To load some data:

>>> from organoid_tracker.core.experiment import Experiment
>>> experiment = Experiment()
>>> from organoid_tracker.guizela_tracker_compatibility import guizela_data_importer
>>> guizela_data_importer.add_data_to_experiment(experiment, "path/to/directory")

And to save the data:

>>> from organoid_tracker.core.experiment import Experiment
>>> experiment = Experiment()  # Placeholder, replace with actual data
>>> from organoid_tracker.guizela_tracker_compatibility import guizela_data_exporter
>>> guizela_data_exporter.export_links(experiment, "path/to/directory")
"""
