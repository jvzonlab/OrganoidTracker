"""Predictions particle positions using an already-trained convolutional neural network."""

from autotrack.config import ConfigFile
from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.resolution import ImageResolution
from autotrack.imaging import general_image_loader, io
from autotrack.position_detection_cnn import predicter
from autotrack.position_detection_cnn.background_supressing_image_loader import BackgroundSupressingImageLoader

experiment = Experiment()

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_positions")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of TIFF files, please paste the path here."
                                                       " Else, if you have a LIF file, please paste the path to that"
                                                       " file here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for three digits "
                                                        "representing the time point)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
try:
    experiment.images.resolution()
except UserError:
    # Need to read resolution
    _pixel_size_x_um = float(config.get_or_prompt("pixel_size_x_um", "What is the resolution in the x and y direction? Please enter a number (μm/px)."))
    _pixel_size_z_um = float(config.get_or_prompt("pixel_size_z_um", "What is the resolution in the z direction? Please enter a number (μm/px)."))
    _time_point_duration_m = float(config.get_or_prompt("time_point_duration_m", "How often are the images taken? Please specify the number of minutes."))
    resolution = ImageResolution(_pixel_size_x_um, _pixel_size_x_um, _pixel_size_z_um, _time_point_duration_m)
    experiment.images.set_resolution(resolution)

_checkpoint_folder = config.get_or_prompt("checkpoint_folder", "Please paste the path here to the \"checkpoints\" folder containing the trained model.")
_output_file = config.get_or_default("positions_output_file", "Automatic positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_debug_folder = config.get_or_default("predictions_output_folder", "", comment="If you want to see the raw prediction images, paste the path to a folder here. In that folder, a prediction image will be placed for each time point.")
if len(_debug_folder) == 0:
    _debug_folder = None
config.save_if_changed()
# END OF PARAMETERS

experiment.images.image_loader(experiment.images.image_loader())
if not experiment.images.image_loader().has_images():
    print("No images were found. Please check the configuration file and make sure that you have stored images at"
          " the specified location.")
    exit(1)

print("Using neural networks to predict positions...")
positions = predicter.predict(experiment.images, _checkpoint_folder, split=True, out_dir=_debug_folder)
experiment.positions.add_positions_and_shapes(positions)

print("Saving file...")
io.save_data_to_json(experiment, _output_file)
