# API reference
[← Back to main page](INDEX.md)

OrganoidTracker contains many functions for working with experimental data. Those functions should make it possible to plot useful information. You can use these functions from standalone Python scripts, from [plugins](PLUGIN_TUTORIAL.md) or from [Jupyter Notebooks](JUPYTER_NOTEBOOK.md).

Note: any method, function and field that has a name starting with an underscore (`_`) should not be used by external code. Ask if there is an alternative way to do it.

Note: for a complete overview of all methods and properties in the `Experiment` object, use the `Help -> Show data inspector...` menu option. To view a (daunting) overview of all classes and methods in OrganoidTracker, run `pydoc -b` from the command line, while you are in the OrganoidTracker folder and in the OrganoidTracker conda environment.

### How do I save and load tracking data?

Saving and loading should be straightforward. The function `io.load_data_file` can load any [supported tracking format](TRACKING_FORMATS.md). The function `io.save_data_to_json` can just save to the standard data format.

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import io

# Creating a new experiment without any data
experiment = Experiment()

# Loading a new experiment from existing data
experiment = io.load_data_file("my_file_name.aut")

# Saving
io.save_data_to_json(experiment, "my_file_name.aut")
```

For other file formats, you need to write a script yourself. See the [custom tracking formats](CUSTOM_TRACKING_FORMATS.md) page for an example.

### How do I iterate over all positions of a particular time point?

If you want to get the detected positions on a certain time point, you can do it like this:

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core import TimePoint
experiment = Experiment()

time_point = TimePoint(2)  # This represents the second time point of the experiment
positions = experiment.positions.of_time_point(time_point)

print("Found positions:", positions)
```

### How do I find the first/last time point?

Do you mean the first/last time point with images, the first/last with positions, or the first/last in general? Here's how you would get the last time point number:

```python
from organoid_tracker.core.experiment import Experiment

experiment = Experiment()
last_number = experiment.positions.last_time_point_number()  # Last time point for which we have positions
last_number = experiment.images.last_time_point_number()  # Last time point number for which we have images
last_number = experiment.last_time_point_number()  # Highest of the above
```

For getting the first time point instead of the last, write `first` instead of `last`. If no data exists in the entire experiment, then these functions simply return `None`.

Note: these functions return an `int`. To convert that to a `TimePoint` instance, use `time_point = TimePoint(number)`

### How do I iterate over all positions in all time points?
If you want to loop through all positions of all time points in an experiment, you can do that as follows:

```python
from organoid_tracker.core.experiment import Experiment
experiment = Experiment()

for time_point in experiment.time_points():
    positions_of_time_point = experiment.positions.of_time_point(time_point)
    print("In time point", time_point.time_point_number(), "there are",
          len(positions_of_time_point), "time points.")
```

### How do I find the nearest position?

If you want to find the nearest detected position from a set of positions, there are a few pre-made functions for that. For example, this is how to get the nearest four positions around a position at (x, y, z) =  (15, 201, 3):

```python
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.position import Position
from organoid_tracker.linking import nearby_position_finder

positions = set()  # This should be list of positions, see above how to get them
image_resolution = ImageResolution(0.32, 0.32, 2, 12)  # Translation of px to um

around_position = Position(x=15, y=201, z=3)
nearby_position_finder.find_closest_n_positions(positions, around=around_position, max_amount=4, resolution=image_resolution)
```

There are a few other functions:

```python
from organoid_tracker.linking import nearby_position_finder

# Finds a single closest position
nearby_position_finder.find_closest_position(..., around=...)

# Finds the closest position, as well as positions up to N times away as the
# closest position
N = 2
nearby_position_finder.find_close_positions(..., around=..., tolerance=N)
```

### How do I check whether a link exists between two positions?
The connections between positions at different time points are called links. This is how you can check if two positions have a link between each other:

```python
from organoid_tracker.core.experiment import Experiment
experiment = Experiment()
position_a = ...
position_b = ...

if experiment.links.contains_link(position_a, position_b):
    ... # There is a link
else:
    ... # There is no link
```

Note: this method only returns True if there is a *direct* link between the two positions, so if they are in consecutive time points.

### How do I get the position of the same cell in the next/previous time point?
You can get find out to which position a position is connected using the `find_pasts` and `find_futures` methods.

```python
from organoid_tracker.core.experiment import Experiment
experiment = Experiment()
position = ...

future_positions = experiment.links.find_futures(position)
```

The resulting set will usually have a size of 1, as it just returns the position of the position one time point later. However, if a cell dies or goes out of view, the set of future positions will be empty. If the position was in the last time point of an experiment then the set of future positions will be empty as well. In contrast, if a cell divides, the set will have two elements.

The set of past positions will usually have a size of 1. A size of 0 occurs if the position just went into the view in this time point, or if the time point is the first time point of the experiment. A size of 2 would indicate that two cells merged into one cell. (However, please note that the lineage tree scripts in OrganoidTracker cannot correctly draw this.)

### How do I measure some property of a cell over time?
It's easiest to run backwards in time. If you would run forwards, then it's not clear what should happen when a cell divides. What daughter should then be followed?

See above for how to get positions for a particular time point, and how to get the last time point of an experiment. Once you have somehow obtained the last position, you can run back in time as follows:

```python
from organoid_tracker.core.experiment import Experiment
experiment = Experiment()
last_position = ...

for position in experiment.links.iterate_to_past(last_position):
    # Record some state of the position, like
    x_location = position.x
    time_point_number = position.time_point_number()
```

### How do I find all dead cells?
A cell can die within the organoid epithelium, but it is also possible that a live cell was extruded. Although both events will cause the demise of the cell, OrganoidTracker still makes a difference between the two.

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.linking_analysis import linking_markers

experiment = Experiment()
shed_cells = linking_markers.find_shed_positions(experiment.links, experiment.position_data)
dead_cells = linking_markers.find_death_positions(experiment.links, experiment.position_data)
dead_and_shed_cells = linking_markers.find_death_and_shed_positions(experiment.links, experiment.position_data)
```

You can then iterate over those using a loop: `for position in dead_cells:`.

### How do I work with lineage trees?
Imagine a lineage tree like this:

```
   |  (A)            |  (B)
   |                 |
-------              |
|      |             |
|      |           -----
|    -----         |   |
|    |   |         |   |
|    |   |         |   |
```

In OrganoidTracker, this lineage tree would be represented by five so-called tracks. A track is a sequence of cell positions. Once a cell divides, two new tracks are started. Therefore, every biological cell cycle is represented by a single track. The above lineage trees contain 5 tracks in lineage A and 3 tracks in lineage B.

You can of course extract all necessary information from the `find_next` and `find_futures` methods discussed above. But it is faster for the computer to quickly jump to the end of the track, than walking through all links time point for time point.

You can get the track a position belongs to using the following method:

```python
from organoid_tracker.core.experiment import Experiment
experiment = Experiment()
position = ...

track = experiment.links.get_track(position)

print("Track goes from time point", track.min_time_point_number(), "to",
      track.max_time_point_number(), "after which", len(track.get_next_tracks()),
      "directly follow")
```

The `track.get_next_tracks()` method never returns only one track. It returns either zero tracks (if the lineage ended there) or two tracks (if the cell divided), representing the tracks of a daughter cell.

If it returns zero tracks, then either the cell died, or it went out of the view. You can check for a cell death like this:

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker

experiment = Experiment()
position = ...

track = experiment.links.get_track(position)
end_marker = linking_markers.get_track_end_marker(experiment.position_data, track.find_last_position())
if end_marker == EndMarker.DEAD:
    # Cell died
    ...
else:
    # Cell went out of the view
    ...
```

### How do I know how much time a time point takes?
The time between two time points is defined by the time resultion:

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.resolution import ImageResolution

# Normally, you would load an experiment that has the resultion already stored.
# Here, we just set a resolution: (x in um, y, z, t in minutes)
experiment = Experiment()
experiment.images.set_resolution(ImageResolution(0.32, 0.32, 2, 12))

# Here's how to get the time between time points:
minutes_between_time_points = experiment.images.resolution().time_point_interval_m
hours_between_time_points = experiment.images.resolution().time_point_interval_h
```

### How do I get/set the image resolution?
The images resolution is accessed as follows:

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.resolution import ImageResolution

experiment = Experiment()

# This is how you define the resolution (x in um, y, z, t in minutes)
experiment.images.set_resolution(ImageResolution(0.32, 0.32, 2, 12))
# (if you load an expeirment from an AUT file, the resolution will
# likely be defined already, and you don't need the above line)

# Here's how to get the size of a pixel in micrometers (um)
minutes_between_time_points = experiment.images.resolution().time_point_interval_m
hours_between_time_points = experiment.images.resolution().time_point_interval_h
```


### How do I automatically open OrganoidTracker from a standalone script?
You can of course save data files and then manually open them. However, you can also directly open the visualizer from a script, with your data already loaded. Say, you have an image that you want to display:

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.gui import launcher
from organoid_tracker.image_loading.array_image_loader import SingleImageLoader
from organoid_tracker.visualizer import standard_image_visualizer

array = ... # Some single color 3D numpy array, representing an image

# Set up Images object
images = Images()
images.image_loader(SingleImageLoader(array))
# ... you could also load LIF files, TIFF files, set a resolution, etc.

# Set up Experiment object, holding the given images
experiment = Experiment()
experiment.images = images
# ... you can add any data you want to the experiment

# Open the visualizer
standard_image_visualizer.show(experiment)
launcher.mainloop()  # Necessary! Google "event loop" for details
```
