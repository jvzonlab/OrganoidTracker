# Detects cell positions using local maxima in a distance transform image.
# First, a microscopy image is converted to purely black-or-white using direct thresholding
# Then a distance transform is applied, such that pixels further from the background have a higher intensity.
# Then the local maximums become the particles

from imaging import tifffolder, Experiment, io
from particle_detection import detection_visualizer
import particle_detection.detector_for_experiment as detector
from matplotlib import pyplot
from os import path

from particle_detection.dt_detection import DistanceTransformDetector
from particle_detection.dt_with_laplacian import DistanceTransformWithLaplacianDetector


# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_frame = 1
_max_frame = 5000
_method = DistanceTransformDetector()
_method_parameters = {
    "min_intensity": 0.6,  # Intensities below this value are considered to be background
    "max_cell_height": 8
}
# File is loaded if it exists, otherwise it created
_particles_file = "../Results/" + _name + "/" + detector.get_file_name(_method, _method_parameters)
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_frame=_min_frame, max_frame=_max_frame)
if path.exists(_particles_file):
    io.load_positions_from_json(experiment, _particles_file)
else:
    detector.detect_particles_in_3d(experiment, _method, **_method_parameters)
    io.save_positions_to_json(experiment, _particles_file)

detection_visualizer.show(experiment, _method, _method_parameters)

print("Done!")
pyplot.show()
