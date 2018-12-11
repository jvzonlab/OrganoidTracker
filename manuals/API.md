# API reference
Autotrack contains many functions for working with experimental data.
Those functions should make it possible to plot useful information.

Parts of Autotrack are pretty general, and can be used for any kind of particle. Other
parts are specialized towards biological cells. Each cell is then represented as a single
particle.

Note: any method, function and field that has a name starting with an underscore (`_`)
should not be used by external code. Ask if there is an alternative way to do it.

## Saving and loading data

We have different data file formats. One is the original file format used by Guizela,
another is my file format, which stores everything in a single file.

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

## Finding particles

If you want to get the particles of a certain time point, you can do it like this:

```python
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
experiment = Experiment()

time_point = TimePoint(2)  # This represents the second time point of the experiment
particles = experiment.particles.of_time_point(time_point)

print("Found particles:", particles)
```

Or if you want to loop through all time points in an experiment:

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()

for time_point in experiment.time_points():
    particles_of_time_point = experiment.particles.of_time_point(time_point)
    print("In time point", time_point.time_point_number(), "there are",
          len(particles_of_time_point), "time points.")
```

If you want to find the nearest particle from a set of particles, there are a few
pre-made functions for that. For example, this is how to get the nearest four particles
around a particle at (x, y, z) =  (15, 201, 3):

```python
from autotrack.core.particles import Particle
from autotrack.linking import nearby_particle_finder
particles = set()  # This should be list of particles, see above how to get them

around_particle = Particle(x=15, y=201, z=3)
nearby_particle_finder.find_closest_n_particles(particles, around_particle, max_amount=4)
```

There are a few other functions:

```python
from autotrack.linking import nearby_particle_finder

# Finds a single closest particle
nearby_particle_finder.find_closest_particle(..., around=...)

# Finds the closest particle, as well as particles up to N times away as the
# closest particle
N = 2
nearby_particle_finder.find_close_particles(..., around=..., tolerance=N)
```

## Working with links
The connections between particles at different time points are called links. This is how
you can check if two particles have a link between each other:

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()
particle_a = ...
particle_b = ..

if experiment.links.contains_link(particle_a, particle_b):
    ... # There is a link
else:
    ... # There is no link
```

Note: this method only returns True if there is a *direct* link between the two particles,
so if they are in consecutive time points.

You can get find out to which particle a particle is connected using the `find_pasts`
and `find_futures` methods.

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()
particle = ...

future_particles = experiment.links.find_futures(particle)
```

The resulting set will usually have a size of 1, as it just returns the position of the
particle one time point later. However, if a cell dies or goes out of view, the set of
future particles will be empty. If the particle was in the last time point of an
experiment then the particle will also have no future particles connected. If a cell
divides, the set will have two elements.

The set of past positions will usually have a size of 1. A size of 0 occurs if the
particle just went into the view in this time point, or if the time point is the first
time point of the experiment.

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

In Autotrack, this lineage tree would be represented by five so-called tracks. A track is
a sequence of cell positions. Once a cell divides, two new tracks are started. Therefore,
every biological cell cycle is represented by a single track. The above lineage trees
contain 5 tracks in lineage A and 3 tracks in lineage B.

You can of course extract all necessary information from the `find_next` and
`find_futures` methods discussed above. But it is faster for the computer to quickly jump
to the end of the track, than walking through all links time point for time point.

You can get the track a particle belongs to using the following method:

```python
from autotrack.core.experiment import Experiment
experiment = Experiment()
particle = ...

track = experiment.links.get_track(particle)

print("Track goes from time point", track.min_time_point_number(), "to",
      track.max_time_point_number(), "after which", len(track.get_next_tracks()),
      "directly follow")
```

The `track.get_next_tracks()` method never returns only one track. It returns either zero
tracks (if the lineage ended there) or two tracks (if the cell divided), representing the
tracks of a daughter cell.
