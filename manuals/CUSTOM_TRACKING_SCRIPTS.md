# Custom scoring system
[← Back to main page](index.md)

If you use the cell tracker in the [standard way](AUTOMATIC_TRACKING.md), you would first obtain a set of positions using an U-net type neural network, and then add likelihoods for divisions and linking. Each of these steps can be replaced by an alternative method, if you desire. This does require some programming.

## Creating a custom position detector
If you create a custom position detector, you no longer use the `organoid_tracker_predict_positions.py` script. Instead, you would use your own custom script.

An alternative approach is to keep using the OrganoidTracker script, but with a custom trained network, so that it works better with your images. See [this page](TRAINING_THE_NETWORK.md) for more information.

This guide is about how to replace the OrganoidTracker script with a custom script. The guide does not explain how to go from images to a list of positions, since that is an extremely broad topic. Instead, we will assume that you have already found some way to detect positions, and simply want to put that in a format that OrganoidTracker understands.

You can write positions to an `.aut` file as follows:

```python
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.images import ImageResolution
from organoid_tracker.imaging import io

# This creates an empty experiment
experiment = Experiment()

# Next, we add some positions
# (all in pixel coordinates)
experiment.positions.add(Position(0, 2, 3, time_point_number=1))
experiment.positions.add(Position(3, 3, 2, time_point_number=2))
# ... etc

# The linking step of the cell tracker requires setting a resolution, so we can set that too
# The numbers are x (um/px), y (um/px), z (um/px), t (minutes)
experiment.images.set_resolution(ImageResolution(0.32, 0.32, 2, 12))

# Finally, we save the file
io.save_data_to_json(experiment, "output_file.aut")
```

That's it! Now you end up with a file that OrganoidTracker can read. You could use this file as the input file for the `organoid_tracker_predict_divisions.py` and `organoid_tracker_predict_links,py` scripts.


## Creating a custom linking/division scoring system
Here, we want to replace the `organoid_tracker_predict_links.py` and the `organoid_tracker_predic_positions.py` scripts.

The linking step requires all possible links to be present, and every one of those possible links to have a link penalty. In addition, every position must have a division penalty set.

The link/division penalty is the log likelihood of the chance of that link being real:

```
penalty = -log10(chance) + log10(1 - chance)
```

You can specify the set of possible links using the `add_link` function:

```python
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io

# Load some tracking data
experiment = io.load_data_file("input file.aut")

# Select some positions
# (normally you would retrieve those from experiment.positions instead of hardcoding)
position1 = Position(0, 0, 0, time_point_number=0)
position2 = Position(1, 2, 1, time_point_number=1)
position3 = Position(1, 9, 0, time_point_number=1)

# Specify the possible links
# In this case, position1 is linked to position2 and/or position3
experiment.links.add_link(position1, position2)
experiment.links.add_link(position1, position3)
```

There's a premade function in OrganoidTracker that select the nearest position as the potenial candidate for a link, as well as any other position that is at most twice as far away. In our experience, this criterium works very well to select all possible links.

You can call that function as follows:

```python
from organoid_tracker.imaging import io
from organoid_tracker.linking import nearest_neighbor_linker

# Load some tracking data
experiment = io.load_data_file("input file.aut")

experiment.links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
```

Now that you have your set of links, you can score each possible link as follows:

```python
from organoid_tracker.imaging import io
from organoid_tracker.linking import nearest_neighbor_linker
import math

# Load the experiment, create possible links
experiment = io.load_data_file("input file.aut")
experiment.links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)

# Loop over all links
for source, target in experiment.links.find_all_links():
    chance = 0.55  # Replace this with the chance of this link being real
    # The chance may not be 0 or 1, so limit it from 0.00000001 to 0.99999999

    penalty = -math.log10(chance)+math.log10(1-chance)
    experiment.link_data.set_link_data(source, target, "link_penalty", penalty)
```

For divisions, you would use a very similar approach. You specify, for each nucleus position, the chance of that nucleus having divided into two nuclei in the next time point.

```python
from organoid_tracker.core.experiment import Experiment
import math

# Load the experiment, create possible links
experiment = Experiment()

# Loop over all links
for position in experiment.positions:
    chance = 0.05  # Replace this with the chance of this position being a cell that divides
    # The chance may not be 0 or 1, so limit it from 0.00000001 to 0.99999999

    penalty = -math.log10(chance)+math.log10(1-chance)
    experiment.position_data.set_position_data(position, "division_penalty", penalty)
```

That's it! Don't forget to save your experiment at the end of the script with `io.save_data_to_json(experiment, "output_file.aut")`. Now your tracking data file is ready for the `organoid_tracker_create_links.py` script.

## Creating a custom linking script
The linking script is already highly customizable, as you can chance which positions, links and scores it receives.

But if you want to replace it anyways, that's possible. See the `organoid_tracker_create_nearest_neigbhors.py` script for an example. This script simply always links positions to the nearest position.
