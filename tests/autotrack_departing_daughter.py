import math

from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
import matplotlib.pyplot as plt
import numpy as np

def is_symmmetric(links, track1, track2):
    """Returns True if symmetric (both divide or both don't divide), False otherwise."""
    ...


# Loading a new experiment from existing data
experiment = io.load_data_file(
    "S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/31-1_correctdata.aut")

# Store mothers cell
mothers = cell_division_finder.find_mothers(experiment.links)

# Empty list for distance
distance_list = []
distance_36min_list = []

# Get position and distance for every mother cells and their daughters
for mother in mothers:
    daughter1, daughter2 = experiment.links.find_futures(mother)
    distance = daughter1.distance_squared(daughter2)
    distance_sqrt = math.sqrt(distance)
    # Check lineage for distance less than 35
    if distance_sqrt < 35:
        track_1 = experiment.links.get_track(daughter1)
        track_2 = experiment.links.get_track(daughter2)
        if is_symmmetric(experiment.links, track_1, track_2):
            ...
        else:

        print("Track goes from time point", track_1.min_time_point_number(), track_2.min_time_point_number(), "to",
              track_1.max_time_point_number(), track_2.max_time_point_number(), "after which",
              len(track_1.get_next_tracks()), len(track_2.get_next_tracks()),
              "directly follow")
    # Get position and distance of daughter cells if the distance less than  200*2, and only execute when it is 0 or more than 1
    while distance < 200**2:
        next_daughters1 = experiment.links.find_futures(daughter1)
        next_daughters2 = experiment.links.find_futures(daughter2)
        if len(next_daughters1) != 1 or len(next_daughters2) != 1:
            break
        daughter1 = next_daughters1.pop()
        daughter2 = next_daughters2.pop()
        distance = daughter1.distance_squared(daughter2)
        # Compare the distance of daughter cells in different time point
        if daughter1.time_point_number() == mother.time_point_number() + 10:
            distance_list.append(distance_sqrt)
            distance_36min_list.append(distance_36mins)
        if daughter1.time_point_number() == mother.time_point_number() +3:
            distance_36mins = distance_sqrt




# .. Loop has ended, now our list is complete
# Make plot for distance comparison
plt.scatter(distance_list, distance_36min_list)
plt.xlabel('distance between sister cells after 2 hours')
plt.ylabel('distance between sister cells after 36 mins')
plt.suptitle('Sister cells distances after 2 hours v. 36 mins')
#plt.show()
