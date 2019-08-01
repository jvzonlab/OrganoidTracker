"""Tries to find out which image loader to use based on two settings, container and patter
n."""
import os.path

from ai_track.core.experiment import Experiment
from ai_track.image_loading.noise_suppressing_filter import NoiseSuppressingFilter


def load_images(experiment: Experiment, container: str, pattern: str,
                min_time_point: int = 0, max_time_point: int = 1000000000):
    """Loads images from any of the supported formats. The container is a file or a directory, pattern is the format
    used to search within that file or directory. For a sequence of TIFF files, container will be a directroy and
    pattern the pattern of files in that directory. For a LIF file, container will be the LIF file, and pattern the name
    of the experiment within that file. Etc."""
    if container.endswith(".lif"):  # Try as LIF file
        from ai_track.image_loading import liffile_image_loader
        liffile_image_loader.load_from_lif_file(experiment.images, container, pattern, min_time_point, max_time_point)
        return
    if container.endswith(".nd2"):
        from ai_track.image_loading import nd2file_image_loader
        nd2file_image_loader.load_image_series_from_config(experiment.images, container, pattern, min_time_point, max_time_point)
        return
    if os.path.isdir(container):  # Try as TIF folder
        from ai_track.image_loading import folder_image_loader
        folder_image_loader.load_images_from_folder(experiment, container, pattern, min_time_point, max_time_point)
        return
    raise ValueError("Unknown file format: " + container + " " + pattern)
