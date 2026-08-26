"""For training the linking network, we not only need positive examples (which are provided by the competition), but
 also negative examples. This script uses the position predictions, and add them to every other time point of the
 training data.

(We can't insert the positions every time point - then we are inserting a ton of false negatives, as many cells that
are actually linked won't have a link then.)
"""
import os

from tqdm import tqdm

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import io, list_io
from organoid_tracker.linking import nearby_position_finder

_LINKS_TRUE_POSITIVE_FILE = r"D:\Nextcloud\rkok\Biohub competition\Official data\Training tracks Rutger.autlist"
_POSITION_PREDICTIONS_FILE = r"D:\Nextcloud\rkok\Biohub competition\Positions predictions (attempt 1)\Automatic positions\_All.autlist"
_OUTPUT_FOLDER = r"E:\Scratch\Biohub competition\Linking network training data"

# Positions too close to the ground truth positions are not added, as they are likely the same cell
# Positions too far from the ground truth positions are not added either, as they are not interesting negative examples to train on
_MIN_DISTANCE_FROM_GROUND_TRUTH_UM = 5
_MAX_DISTANCE_FROM_GROUND_TRUTH_UM = 20


def _name(experiment: Experiment) -> str:
    """Removes the "Hard" prefix sometimes used, so that we can properly match"""
    return experiment.name.get_name().replace("Hard ", "")


def main():
    autlist_file = os.path.join(_OUTPUT_FOLDER, "All.autlist")
    if os.path.exists(autlist_file):
        os.remove(autlist_file)

    # Load all position predictions (without links)
    print("Loading position predictions...")
    all_position_predictions = dict()
    for position_predictions in list_io.load_experiment_list_file(_POSITION_PREDICTIONS_FILE, load_images=False):
        all_position_predictions[_name(position_predictions)] = position_predictions
    print(all_position_predictions.keys())

    # Load all ground truth experiments and add the position predictions to every other time point
    for experiment in list_io.load_experiment_list_file(_LINKS_TRUE_POSITIVE_FILE):
        position_predictions = all_position_predictions.get(_name(experiment))
        if position_predictions is None:
            print(f"No position predictions found for experiment: {experiment.name}")
            continue
        print(experiment.name)

        resolution = experiment.images.resolution()

        # Run through every other time point and add the position predictions to the ground truth experiment,
        # if they are not too close to existing ground truth positions
        for time_point in experiment.positions.time_points():
            if time_point.time_point_number() % 2 != 0:
                continue

            existing_positions = experiment.positions.of_time_point(time_point)

            # Add the position predictions to the experiment
            prediction_offset = position_predictions.images.offsets.of_time_point(time_point)
            our_offset = experiment.images.offsets.of_time_point(time_point)
            for position in position_predictions.positions.of_time_point(time_point):
                position -= prediction_offset
                position += our_offset

                closest_ground_truth_position = nearby_position_finder.find_closest_position(existing_positions, around=position, resolution=resolution, max_distance_um=_MAX_DISTANCE_FROM_GROUND_TRUTH_UM)
                if closest_ground_truth_position is None:
                    continue  # Too far from any existing ground truth position, so we don't add it
                distance_squared = closest_ground_truth_position.distance_squared(position, resolution=resolution)
                if distance_squared < _MIN_DISTANCE_FROM_GROUND_TRUTH_UM ** 2:
                    continue  # Too close to an existing ground truth position, so we don't add it

                experiment.positions.add(position)

        io.save_data_to_json(experiment, os.path.join(_OUTPUT_FOLDER, experiment.name.get_name() + "." + io.FILE_EXTENSION))
        list_io.save_experiment_list_file([experiment], autlist_file, append_to_file=True)


if __name__ == "__main__":
    main()
