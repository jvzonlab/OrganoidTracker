import os.path

from autotrack.core.experiment import Experiment


def load_images(experiment: Experiment, path: str, format: str,
                min_time_point: int = 0, max_time_point: int = 1000000000):
    """Loads images from any of the supported formats. Path is a file or a directory, format is the format used to
    search within that file or directory."""
    if path.endswith(".lif"):  # Try as LIF file
        from autotrack.imaging import liffile
        liffile.load_from_lif_file(experiment.images, path, format, min_time_point, max_time_point)
        return
    if os.path.isdir(path) and ".tif" in format:  # Try as TIF folder
        from autotrack.imaging import tifffolder
        tifffolder.load_images_from_folder(experiment, path, format, min_time_point, max_time_point)
        return
    raise ValueError("Unknown file format: " + path + " " + format)
