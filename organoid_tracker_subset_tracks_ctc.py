"""Predicts cell positions using an already-trained convolutional neural network."""
from skimage.feature import peak_local_max
from tifffile import tifffile

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io
from organoid_tracker.linking.nearby_position_finder import find_closest_n_positions

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("subset_tracks_ctc")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_positions_file = config.get_or_prompt("positions_file",
                                            "Where are the cell postions saved?")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

_start_positions_image = config.get_or_prompt("start_positions_image", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.")
config.save_and_exit_if_changed()
# END OF PARAMETERS

experiment = io.load_data_file(_positions_file, _min_time_point, _max_time_point)
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

# Check if images were loaded
if not experiment.images.image_loader().has_images():
    print("No images were found. Please check the configuration file and make sure that you have stored images at"
          " the specified location.")
    exit(1)

start_positions = PositionCollection()

im = tifffile.imread(_start_positions_image)

coordinates = peak_local_max(im, min_distance=2, threshold_abs=0.1,
                             exclude_border=False)

for coordinate in coordinates:
    print(coordinate)
    pos = Position(coordinate[2], coordinate[1], coordinate[0],
                   time_point=TimePoint(0))
    start_positions.add(pos)

start_experiment = Experiment()
start_experiment.positions.add_positions(start_positions)
io.save_data_to_json(start_experiment, 'start_positions.aut')


subset = []

for pos in start_positions:
    nearest_neighbor = find_closest_n_positions(experiment.positions.of_time_point(TimePoint(0)), around= pos, max_amount=1,
                                                resolution=experiment.images.resolution())

    track = experiment.links.get_track(list(nearest_neighbor)[0])
    print(nearest_neighbor)
    print(track)
    all_tracks = track.find_all_descending_tracks(include_self=True)

    for track in all_tracks:
        subset = subset + list(track.positions())

print('remove')
subset = set(subset)
print(subset)
print(experiment.positions)
to_remove = subset.difference(set(experiment.positions))
to_remove= set(experiment.positions).difference(subset)

print(to_remove)
print(len(to_remove))


experiment.remove_positions(to_remove)

print("Saving file...")
io.save_data_to_json(experiment, 'subsetted_positions.aut')


