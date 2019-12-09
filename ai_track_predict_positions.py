"""Predictions particle positions using an already-trained convolutional neural network."""

from ai_track.config import ConfigFile, config_type_bool, config_type_int, config_type_float
from ai_track.core.experiment import Experiment
from ai_track.imaging import io
from ai_track.image_loading import general_image_loader
from ai_track.position_detection_cnn import predicter

experiment = Experiment()

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_positions")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

_checkpoint_folder = config.get_or_prompt("checkpoint_folder", "Please paste the path here to the \"checkpoints\" folder containing the trained model.")
_output_file = config.get_or_default("positions_output_file", "Automatic positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_smooth_stdev = config.get_or_default("smooth_stdev", str(1), comment="Standard deviation of Gaussian smooth algorithm."
                                      " Used to improve peak finding.", type=config_type_int)
_predictions_threshold = config.get_or_default("predictions_threshold", str(0.1), comment="Prediction peaks with"
                                               " values less than this (on a scale of 0 to 1) are ignored.",
                                               type=config_type_float)
_split = config.get_or_default("save_video_ram", "true", comment="Whether video RAM should be saved by splitting"
                                                                 " the images into smaller parts, and processing"
                                                                 " each part independently.", type=config_type_bool)
_debug_folder = config.get_or_default("predictions_output_folder", "", comment="If you want to see the raw prediction images, paste the path to a folder here. In that folder, a prediction image will be placed for each time point.")
if len(_debug_folder) == 0:
    _debug_folder = None
config.save()
# END OF PARAMETERS

if not experiment.images.image_loader().has_images():
    print("No images were found. Please check the configuration file and make sure that you have stored images at"
          " the specified location.")
    exit(1)

print("Using neural networks to predict positions...")
positions = predicter.predict(experiment.images, _checkpoint_folder, split=_split, out_dir=_debug_folder,
                              smooth_stdev=_smooth_stdev, predictions_threshold=_predictions_threshold)
experiment.positions.add_positions_and_shapes(positions)

print("Saving file...")
io.save_data_to_json(experiment, _output_file)
