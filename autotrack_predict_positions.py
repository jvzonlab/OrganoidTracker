"""Predictions particle positions using an already-trained convolutional neural network."""

from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.core.resolution import ImageResolution
from autotrack.imaging import general_image_loader, io
from autotrack.imaging.limited_z_image_loader import LimitedZImageLoader
from autotrack.position_detection_cnn import predicter

experiment = Experiment()

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_positions")
_images_folder = config.get_or_default("images_folder", "../", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for three digits "
                                                        "representing the time point)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
try:
    experiment.images.resolution()
except ValueError:
    # Need to read resolution
    _pixel_size_x_um = float(config.get_or_default("pixel_size_x_um", str(0.32), store_in_defaults=True))
    _pixel_size_y_um = float(config.get_or_default("pixel_size_y_um", str(0.32), store_in_defaults=True))
    _pixel_size_z_um = float(config.get_or_default("pixel_size_z_um", str(2), store_in_defaults=True))
    _time_point_duration_m = float(config.get_or_default("time_point_duration_m", str(12), store_in_defaults=True))
    resolution = ImageResolution(_pixel_size_x_um, _pixel_size_y_um, _pixel_size_z_um, _time_point_duration_m)
    experiment.images.set_resolution(resolution)

_checkpoints_folder = config.get_or_default("checkpoints_folder", "checkpoints")
_output_file = config.get_or_default("output_file", "Automatic positions.aut")
config.save_and_exit_if_changed()
# END OF PARAMETERS

experiment.images.image_loader(experiment.images.image_loader())
print("Using neural networks to predict positions...")
positions = predicter.predict(experiment.images, _checkpoints_folder, split_in_four=True, out_dir="out")
experiment.positions.add_positions_and_shapes(positions)

print("Saving file...")
io.save_data_to_json(experiment, _output_file)
