import math

from ai_track.imaging import io
from ai_track.linking import cell_division_finder
from ai_track.core.experiment import Experiment
from ai_track.linking_analysis import linking_markers
from ai_track.linking_analysis.linking_markers import EndMarker
from ai_track.core import TimePoint
import matplotlib.pyplot as plt
import numpy as np


# Loading a new experiment from existing data
experiment = io.load_data_file("S:/AMOLF/groups/zon-group/guizela/multiphoton/organoids/17-07-28_weekend_H2B-mCherry/nd799xy20-stacks/Automatic analysis/31-1_correctdata.aut")

# Store mothers cell
mothers = cell_division_finder.find_mothers(experiment.links)

# Empty list
dz_0_list = []
dz_1_list = []
dz_2_list = []
dz_3_list = []
dz_4_list = []
dz_5_list = []
dz_6_list = []
dz_7_list = []
dz_0_distance = []
dz_1_distance = []
dz_2_distance = []
dz_3_distance = []
dz_4_distance = []
dz_5_distance = []
dz_6_distance = []
dz_7_distance = []

# Get position and distance for every mother cells and their daughters
for mother in mothers:
    daughter1, daughter2 = experiment.links.find_futures(mother)
    distance = daughter1.distance_squared(daughter2)
    distance_sqrt = math.sqrt(distance)
    distance_um = experiment.images.resolution().pixel_size_x_um * distance_sqrt
    dz = abs(daughter1.z - daughter2.z)
    while True:
                next_daughters1 = experiment.links.find_futures(daughter1)
                next_daughters2 = experiment.links.find_futures(daughter2)
                if len(next_daughters1) != 1 or len(next_daughters2) != 1:
                    break
                daughter1 = next_daughters1.pop()
                daughter2 = next_daughters2.pop()
                distance = daughter1.distance_squared(daughter2)
                # Compare the distance of daughter cells in different time point
                if daughter1.time_point_number() == mother.time_point_number() + 7:
                    if dz == 0:
                        dz_0_list.append(dz)
                        dz_0_distance.append(distance_um)
                    elif dz == 1:
                        dz_1_list.append(dz)
                        dz_1_distance.append(distance_um)
                    elif dz == 2:
                        dz_2_list.append(dz)
                        dz_2_distance.append(distance_um)
                    elif dz == 3:
                        dz_3_list.append(dz)
                        dz_3_distance.append(distance_um)
                    elif dz == 4:
                        dz_4_list.append(dz)
                        dz_4_distance.append(distance_um)
                    elif dz == 5:
                        dz_5_list.append(dz)
                        dz_5_distance.append(distance_um)
                    elif dz == 6:
                        dz_6_list.append(dz)
                        dz_6_distance.append(distance_um)
                    elif dz == 7:
                        dz_7_list.append(dz)
                        dz_7_distance.append(distance_um)



print("avg(dz = 0) = ", np.mean(dz_0_list), np.mean(dz_0_distance))
print("avg(dz = 1) = ", np.mean(dz_1_list), np.mean(dz_1_distance))
print("avg(dz = 2) = ", np.mean(dz_2_list), np.mean(dz_2_distance))
print("avg(dz = 3) = ", np.mean(dz_3_list), np.mean(dz_3_distance))
print("avg(dz = 4) = ", np.mean(dz_4_list), np.mean(dz_4_distance))
print("avg(dz = 5) = ", np.mean(dz_5_list), np.mean(dz_5_distance))
print("avg(dz = 6) = ", np.mean(dz_6_list), np.mean(dz_6_distance))
print("avg(dz = 7) = ", np.mean(dz_7_list), np.mean(dz_7_distance))
