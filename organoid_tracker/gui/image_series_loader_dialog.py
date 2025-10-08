import fnmatch
from typing import Iterable

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.imaging.file_loader import FileLoaderType, FileLoader

def prompt_image_series(file_loaders: Iterable[FileLoader], experiment: Experiment) -> bool:
    """Prompts an image series, and loads it into the experiment. Returns whether anything was loaded."""

    # Check which formats we support, and by which loader
    supported_formats_list = list()
    file_loader_by_pattern = dict()
    for file_loader in file_loaders:
        if file_loader.get_type() != FileLoaderType.IMAGE:
            continue
        supported_formats_list.append((file_loader.get_name(), " ".join(file_loader.get_file_patterns())))
        for pattern in file_loader.get_file_patterns():
            file_loader_by_pattern[pattern] = file_loader

    file_path = dialog.prompt_load_file("Select (first) image file", supported_formats_list)
    if file_path is None:
        return False  # Cancelled

    # Replace the existing experiment with one with the same images, but the new data
    for pattern, handler in file_loader_by_pattern.items():
        if fnmatch.fnmatch(file_path, pattern):
            # Found the right handler
            return handler.load_file_interactive(file_path, into=experiment)
    return False
