import math
import os

from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io

FOLDER = r"D:\Nextcloud\rkok\Biohub competition\Positions predictions (attempt 2)\Cleaned positions"

for file_name in os.listdir(FOLDER):
    if not file_name.endswith(".aut"):  # Adjust the file extension as needed
        continue

    experiment = io.load_data_file(os.path.join(FOLDER, file_name))
    removed_count = 0
    for time_point in experiment.positions.time_points():
        positions = list(experiment.positions.of_time_point(time_point))

        to_delete = list()
        for i, position_i in enumerate(positions):
            for j, position_j in enumerate(positions):
                if i >= j:
                    continue
                distance_dx = abs(position_i.x - position_j.x)
                if distance_dx > 4:
                    continue
                distance_dy = abs(position_i.y - position_j.y)
                if distance_dy > 4:
                    continue
                distance_dz = abs(position_i.z - position_j.z)
                if distance_dz > 2:
                    continue

                # Remove one position
                to_delete.append(position_j)

                # Move the other
                experiment.move_position(position_i, Position((position_i.x + position_j.x) / 2,
                                                              (position_i.y + position_j.y) / 2,
                                                              (position_i.z + position_j.z) / 2,
                                                              time_point_number=position_i.time_point_number()),
                                         update_splines=False)

        # Actually remove
        removed_count += len(to_delete)
        experiment.remove_positions(to_delete, update_splines=True)
    io.save_data_to_json(experiment, os.path.join(FOLDER, file_name))
    print(experiment.name, ": ", removed_count, sep="")
