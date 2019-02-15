# API reference
Autotrack contains many functions for working with experimental data. Those functions should make it possible to plot useful information.

Parts of Autotrack are pretty general, and can be used for any kind of position. Other parts are specialized towards biological cells. Each cell is then represented as a single position.

Note: any method, function and field that has a name starting with an underscore (`_`) should not be used by external code. Ask if there is an alternative way to do it.

## Saving and loading data

We have different data file formats. One is the original file format used by Guizela, another is my file format, which stores everything in a single file.

```python
from autotrack.core.experiment import Experiment
from autotrack.imaging import io

# Creating a new experiment without any data
experiment = Experiment()

# Loading a new experiment from existing data
experiment = io.load_data_file("my_file_name.aut")

# Loading an experiment from Guizela's data
# Make sure that the lineages.p file is still in the same directory
# as the track_XXXXX.p files
experiment = io.load_data_file("path/to/data/lineages.p")

# Saving
io.save_data_to_json(experiment, "my_file_name.aut")
```

## Finding positions

If you want to get the detected positions on a certain time point, you can do it like this:

```python
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
experiment = Experiment()

time_point = TimePoint(2)  # This represents the second time point of the experiment
positions = experiment.positions.of_time_point(time_point)

print("Found positions:", positions)
```

Or if you want to loop through all time points in an experiment:

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()

for time_point in experiment.time_points():
    positions_of_time_point = experiment.positions.of_time_point(time_point)
    print("In time point", time_point.time_point_number(), "there are",
          len(positions_of_time_point), "time points.")
```

If you want to find the nearest detected position from a set of positions, there are a few pre-made functions for that. For example, this is how to get the nearest four positions around a position at (x, y, z) =  (15, 201, 3):

```python
from autotrack.core.position import Position
from autotrack.linking import nearby_position_finder
positions = set()  # This should be list of positions, see above how to get them

around_position = Position(x=15, y=201, z=3)
nearby_position_finder.find_closest_n_positions(positions, around_position, max_amount=4)
```

There are a few other functions:

```python
from autotrack.linking import nearby_position_finder

# Finds a single closest position
nearby_position_finder.find_closest_position(..., around=...)

# Finds the closest position, as well as positions up to N times away as the
# closest position
N = 2
nearby_position_finder.find_close_positions(..., around=..., tolerance=N)
```

## Working with links
The connections between positions at different time points are called links. This is how you can check if two positions have a link between each other:

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()
position_a = ...
position_b = ..

if experiment.links.contains_link(position_a, position_b):
    ... # There is a link
else:
    ... # There is no link
```

Note: this method only returns True if there is a *direct* link between the two positions, so if they are in consecutive time points.

You can get find out to which position a position is connected using the `find_pasts` and `find_futures` methods.

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()
position = ...

future_positions = experiment.links.find_futures(position)
```

The resulting set will usually have a size of 1, as it just returns the position of the position one time point later. However, if a cell dies or goes out of view, the set of future positions will be empty. If the position was in the last time point of an experiment then the position will also have no future positions connected. If a cell divides, the set will have two elements.

The set of past positions will usually have a size of 1. A size of 0 occurs if the position just went into the view in this time point, or if the time point is the first time point of the experiment.

## Working with biological lineage trees
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

In Autotrack, this lineage tree would be represented by five so-called tracks. A track is a sequence of cell positions. Once a cell divides, two new tracks are started. Therefore, every biological cell cycle is represented by a single track. The above lineage trees contain 5 tracks in lineage A and 3 tracks in lineage B.

You can of course extract all necessary information from the `find_next` and `find_futures` methods discussed above. But it is faster for the computer to quickly jump to the end of the track, than walking through all links time point for time point.

You can get the track a position belongs to using the following method:

```python
from autotrack.core.experiment import Experiment
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
from autotrack.core.experiment import Experiment
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker

experiment = Experiment()
position = ...

track = experiment.links.get_track(position)
end_marker = linking_markers.get_track_end_marker(experiment.links, track.find_last_position())
if end_marker == EndMarker.DEAD:
    # Cell died
    ...
else:
    # Cell went out of the view
    ...
```
