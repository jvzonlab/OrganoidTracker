# Detects cell positions using local maxima in a distance transform image.
# First, a microscopy image is converted to purely black-or-white using thresholding
# Then a distance transform is applied, such that pixels further from the background have a higher intensity.
# Then the local maximums become the particles

from imaging import tifffolder, Experiment, io
from particle_detection import detector_for_experiment, detection_visualizer
from matplotlib import pyplot
from os import path

# PARAMETERS
# _positions_file and _links_file are optional: if they point to a file that does not exist, simply no positions or
# links data will be displayed.
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_frame = 1
_max_frame = 50
_particles_file = "../Results/" + _name + "/DTLM cell positions.json"  # Loaded if it exists, otherwise created

_method_parameters = {
    "min_intensity": 0.6, # Intensities below this value are considered to be background
    "max_cell_height": 6
}
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_frame=_min_frame, max_frame=_max_frame)
if path.exists(_particles_file):
    io.load_positions_from_json(experiment, _particles_file)
else:
    detector_for_experiment.detect_particles_using_distance_transform(experiment, **_method_parameters)
    io.save_positions_to_json(experiment, _particles_file)

detection_visualizer.show(experiment, _method_parameters)

print("Done!")
pyplot.show()
