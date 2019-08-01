#!/usr/bin/env python3

"""This script is used to detect the shapes of cells, starting from JSON position files and an image folder. It gives a
single multivariate Gaussian as output for each image. This is more accurate than just fitting a stack of ellipses, as
is done in another script. Unfortunately, this script is also much slower.

Parameters: (all sizes are in pixels)

- threshold_block_size: length and/or height of the 2d square used for the adaptive threshold
- distance_transform_smooth_size: smoothing size (must be odd) used for the distance transform
- gaussian_fit_smooth_size: smoothing size (must be odd) used for fitting Gaussians to the image. The more the image
  already looks like Gaussians, the smaller this size can be
- min_segmentation_distance: used for recognizing cell boundaries in a watershed transform
"""
from ai_track.imaging import io
from ai_track.image_loading import general_image_loader
from ai_track.config import ConfigFile, config_type_int
from ai_track.position_detection import gaussian_detector_for_experiment
from ai_track.core.resolution import ImageResolution

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("detect_gaussian_shapes")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_input_file = config.get_or_default("positions_input_file", "Automatic positions.aut")
_positions_output_file = config.get_or_default("positions_and_fit_output_file", "Gaussian fitted positions.aut")

_threshold_block_size = config.get_or_default("adaptive_threshold_block_size", str(51), comment="Size of the block used"
                                              " for local averaging for thresholding. Must be an odd (not even)"
                                              " number.", type=config_type_int)
_cluster_detection_erosion_rounds = config.get_or_default("cluster_detection_erosion_rounds", str(3), comment="Used to "
                                                          " erode the blobs of foreground, to further separate"
                                                          " different Gaussian functions.", type=config_type_int)
_gaussian_fit_smooth_size = config.get_or_default("gaussian_fit_smooth_size", str(7), comment="The Gaussians are fit to"
                                                  " a smoothed image. This setting controls pixel radius for smoothing."
                                                  " If you have a particulary noisy or high-res image, you will need to"
                                                  " increase this value.", type=config_type_int)
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Loading cell positions...")
experiment = io.load_data_file(_positions_input_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Running detection...")
gaussian_detector_for_experiment.perform_for_experiment(experiment, threshold_block_size=_threshold_block_size,
                                                        gaussian_fit_smooth_size=_gaussian_fit_smooth_size,
                                                        cluster_detection_erosion_rounds=_cluster_detection_erosion_rounds)
print("Saving...")
io.save_data_to_json(experiment, _positions_output_file)
print("Done!")
