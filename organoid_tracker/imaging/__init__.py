"""
Various classes to work with time lapses. Unfortunately, this package has become a bit of a catch-all, and it's not
possible to fix that without breaking all existing code.

To load tracking data from disk:

>>> from organoid_tracker.imaging import io
>>> experiment = io.load_data_file("file_name.aut")

And to save it again:
>>> io.save_data_to_json(experiment, "file_name.aut")

To load a dataset of multiple tracking files plus images:
(create those files in the GUI using File -> Tabs -> Export open tabs)

>>> from organoid_tracker.imaging import list_io
>>> for experiment in list_io.load_experiment_list_file("some_file.autlist"):
>>>    ...  # Do something with the experiment

"""
