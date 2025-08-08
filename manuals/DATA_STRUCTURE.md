# Data Structure of OrganoidTracker
[← Back to main page](index.md)

The data structure of OrganoidTracker is not simply a flat table. Instead, for every time-lapse it holds all data in an instance of the `Experiment` class. This class contains all data of the experiment, such as the positions, links, connections and images. This page gives an overview of the data structure, and explains the design philosophy. There's also a [cookbook-like page](API.md) that shows how to use the data structure in practice, with examples, like how to plot some metadata of a cell over time.

*This page is about the Python data structure that is used in memory. If you're looking for information on the on-disk data format, see the [tracking formats page](TRACKING_FORMATS.md).*

Do not access the private fields (those with names starting with an underscore) directly. You are supposed to access the data through the public methods and fields. This gives OrganoidTracker the ability to evolve its internal data structure over time. For example, the Connections class currently uses the NetworkX graph library internally. However, this is an implementation detail, and you should not access this structure directly. Most interactions will instead be through OrganoidTracker-specific methods such as `experiment.connections.find_connections(pos)`. In this way, OrganoidTracker remains free to switch to another graph library. In case you do need the power of the NetworkX library, there's an escape hatch: the method `experiment.connections.to_networkx_graph()` returns a *copy* of the internal data structure, so you can use all the methods of the library. In the event that OrganoidTracker switches to another graph library, then `to_networkx_graph()` method will be reimplemented to convert the object of that graph library back to NetworkX. In this way, your code will keep working.

When using PyCharm or VS Code as your Python editor, the data structure is quite navigable, thanks to their excellent autocomplete. But without that, your best bet is to (a) switch to those editors, or (b) use the data inspector of the OrganoidTracker GUI (`Help` -> `Show data inspector...`).

The most important members of the data structure are:

* `experiment.positions` - holds the positions.
* `experiment.links` - links over time.
* `experiment.connections` - specifies which cells are neighbors.
* `experiment.images` - the images & their resolution, timestamps, spatial offsets and channel information.

The data structures are described below, along with a highlight of their most imporant methods.

# Positions
Positions are stored per time point instead of in one huge table. This is for performance reasons, it makes lookups and edits faster.

```python
experiment.positions.of_time_point(TimePoint(4))

position = Position(1, 2, 3, time_point=TimePoint(4))
experiment.positions.add(position)  # Add a position to the experiment
experiment.positions.set_position_data(position, "data_key", "data value")  # Set metadata for a position
experiment.positions.get_position_data(position, "data_key")  # Get metadata for a position

# To remove a position, use the base experiment class, which will also remove its links and connections
experiment.remove_position(position)
# Moves a position (like remove_position, this method also updates the links and connections data structures)
experiment.move_position(position_old, position_new)
```

# Links
Normally, each position is connected to one position in the next time point. Upon a cell death, it is connected to zero, and upon a division, to two positions. Originally, OrganoidTracker used to store links like this, at the individual time point level. However, for performance reasons it has switched to a structure composed of so-called `LinkingTrack`s.

A `LinkingTrack` is essentially a list of positions, and corresponds to a vertical line in a lineage tree. Each track specifies both the previous and next tracks. For example, upon a cell division, two next tracks are specified: the tracks of the daughter cells. Cell mergers and multi-divisions (three or more daughters) are also supported by the data structure.

When reading the data, you can choose which abstraction you use. If you need information on every time point, for example to measure the fluorescence over time, you have to iterate at the single time point level. However, if you're interested in cell fate, like how much offspring a cell has, it's faster to iterate at the track level.

```python
# At the track level (read-only)
track = experiment.links.get_track(pos)
experiment.links.find_starting_tracks()  # Tracks without parents
experiment.links.find_all_tracks()  # Includes daughter tracks
track.positions()
track.get_previous_tracks()  # Just direct parent
track.get_next_tracks()  # Just direct offspring
track.find_all_previous_tracks(include_self=True)  # Iterates back further
track.find_all_descending_tracks(include_self=True)  # Iterates to further generations

# At the single-time point level (read and write)
experiment.links.find_pasts(pos)
experiment.links.find_futures(pos)
experiment.links.iterate_to_past(pos)
experiment.links.iterate_to_future(pos)
experiment.links.add_link(pos1, pos2)
experiment.links.remove_link(pos1, pos2)
experiment.links.set_link_data(pos1, pos2, "data_key", "data value")
experiment.links.get_link_data(pos1, pos2, "data_key")
```

# Connections
Connections specify which cells are neighbors. (Assuming spatial neighbors, but you can set up neighbors however you want.) Each time point maintains their own independent neighborhood graph.

```python
from organoid_tracker.core.experiment import Experiment
experiment = Experiment()

experiment.connections.find_connections(pos)
experiment.connections.add_connection(pos1, pos2)
experiment.connections.remove_connection(pos1, pos2)
experiment.connections.contains_connection(pos1, pos2)
experiment.connections.to_networkx_graph(time_point)
experiment.connections.get_data_of_connection(pos1, pos2, "data_key")
experiment.connections.set_data_of_connection(pos1, pos2, "data_key", "data value")
```

# Images
Unlike other data structures, images are not kept in memory. Essentially, OrganoidTracker always functions like the virtual stack feature of ImageJ. To load images, you provide your image loader to `experiment.images.image_loader(...)`. You can implement an `ImageLoader` subclass yourself, or use one of the existing ones. OrganoidTracker automatically wraps your image loader into a caching image loader, which stores a certain number of 2D planes in memory, for faster response times.

When loading your images, you can also provide other information to the images object, like the resolution, timings and channel descriptions.

```
experiment.images.image_loader(...)
experiment.images.get_image(time_point, ImageChannel(index_one=1))  # Object with offset
experiment.images.get_image_stack(time_point, channel)  # Raw 3d numpy array
experiment.images.get_image_slice_2d(time_point, channel, z)  # Raw 2d numpy array
experiment.images.set_channel_description(channel, ChannelDescription(channel_name="mCherry", colormap=matplotlib.colormaps.viridis))
experiment.images.resolution()
experiment.images.set_resolution(ImageResolution(0.32, 0.32, 1, 12))
experiment.images.timings()
```


