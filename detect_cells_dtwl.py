# Detects cell positions using local maxima in a distance transform image.
# First, a microscopy image is converted to purely black-or-white using a more complex thresholding technique
# Then a distance transform is applied, such that pixels further from the background have a higher intensity.
# Then the local maximums become the particles
from os import path

import gui
import particle_detection.detector_for_experiment as detector
from core import Experiment
from imaging import tifffolder, io
from visualizer import detection_visualizer
from particle_detection.dt_with_laplacian import DistanceTransformWithLaplacianDetector

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_images_folder = "../Data/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_time_point = 1
_max_time_point = 5000
_method = DistanceTransformWithLaplacianDetector()
_method_parameters = {
    "min_intensity": 0.6,  # Intensities below this value are considered to be background
    "max_cell_height": 8,
    "max_laplacian": -0.5
}
# File is loaded if it exists, otherwise it created
_particles_file = "../Data/" + _name + "/Automatic analysis/Positions/" + detector.get_file_name(_method, _method_parameters)
# END OF PARAMETERS

print("Starting...")

script_dir = path.dirname(__file__)
experiment = Experiment()
tifffolder.load_images_from_folder(experiment, path.join(script_dir, _images_folder), _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
particle_file_abs = path.join(script_dir, _particles_file)
if path.exists(particle_file_abs):
    io.load_positions_from_json(experiment, particle_file_abs)
else:
    detector.detect_particles_in_3d(experiment, _method, **_method_parameters)
    io.save_positions_to_json(experiment, particle_file_abs)

detection_visualizer.show(experiment, _method, _method_parameters)

print("Done!")
gui.mainloop()
