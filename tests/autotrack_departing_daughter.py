import math

from autotrack.imaging import io
from autotrack.linking import cell_division_finder
from autotrack.core.experiment import Experiment
from autotrack.core import TimePoint
import matplotlib.pyplot as plt
import numpy as np


# Loading a new experiment from existing data
experiment = io.load_data_file(
    "S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/31-1_correctdata.aut")

#print(experiment.last_time_point_number())

mothers = cell_division_finder.find_mothers(experiment.links)
# print (mothers)

# .. Make an empty list here
distance_list = []

for mother in mothers:
    #print("Looking at mother", mother)
    daughter1, daughter2 = experiment.links.find_futures(mother)
    #print("Has daughters", daughter1, daughter2)
    distance = daughter1.distance_squared(daughter2)
    #print(distance)
    #print(daughter1, daughter2)
    while distance < 200**2:
        next_daughters1 = experiment.links.find_futures(daughter1)
        next_daughters2 = experiment.links.find_futures(daughter2)
        if len(next_daughters1) != 1 or len(next_daughters2) != 1:
            break
        daughter1 = next_daughters1.pop()
        daughter2 = next_daughters2.pop()
        distance = daughter1.distance_squared(daughter2)
        distance_sqrt = math.sqrt(distance)
        #print(daughter1, daughter2, distance)
        # mother.time_point_number()
        if daughter1.time_point_number() == mother.time_point_number() + 1:
            distance_list.append(distance_sqrt)
           # if (distance_sqrt) < 25:
               # print('mother',mother,'daughter',daughter1, daughter2,'with distance:', distance_sqrt)
            #else:
               # print('sisters are large')

    #print('end of mother cell')

# .. Loop has ended, now our list is complete
plt.hist(distance_list)
plt.suptitle('Sister cells distances after 108 mins')
plt.xlabel('distance between sister cells after 108 mins')
plt.ylabel('number of cases')
plt.show()
