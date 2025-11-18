import fnmatch
from typing import Iterable, Any, Tuple, Dict, List

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging.file_loader import FileLoaderType, FileLoader, LoadInto


def _find_image_loaders(file_loaders: Iterable[FileLoader]) -> Tuple[Dict[str, FileLoader], List[Tuple[str, str]]]:
    """Finds all file loaders that can load images from the given list. Returns a mapping from file pattern to file
     loader, and a list of supported formats."""
    supported_formats_list = list()
    file_loader_by_pattern = dict()
    for file_loader in file_loaders:
        if file_loader.get_type() != FileLoaderType.IMAGE:
            continue
        supported_formats_list.append((file_loader.get_name(), " ".join(file_loader.get_file_patterns())))
        for pattern in file_loader.get_file_patterns():
            file_loader_by_pattern[pattern] = file_loader
    return file_loader_by_pattern, supported_formats_list


def prompt_image_series(file_loaders: Iterable[FileLoader], experiment: Experiment) -> bool:
    """Prompts an image series, and loads it into the experiment. Returns whether anything was loaded."""

    # Check which formats we support, and by which loader
    file_loader_by_pattern, supported_formats_list = _find_image_loaders(file_loaders)

    file_path = dialog.prompt_load_file("Select (first) image file", supported_formats_list)
    if file_path is None:
        return False  # Cancelled

    # Replace the existing experiment with one with the same images, but the new data
    for pattern, handler in file_loader_by_pattern.items():
        if fnmatch.fnmatch(file_path, pattern):
            # Found the right handler
            into = LoadInto(experiment)
            into.allow_extra_tabs = False
            return handler.load_file_interactive(file_path, into=into)
    return False


def prompt_image_series_multiple(window: Window) -> bool:
    """Prompts an image series, and loads it into the experiment. Returns whether anything was loaded."""

    file_loaders = window.registry.get_registered_file_loaders()

    # Check which formats we support, and by which loader
    file_loader_by_pattern, supported_formats_list = _find_image_loaders(file_loaders)

    file_path = dialog.prompt_load_file("Select (first) image file", supported_formats_list)
    if file_path is None:
        return False  # Cancelled

    # Replace the existing experiment with one with the same images, but the new data
    for pattern, handler in file_loader_by_pattern.items():
        if fnmatch.fnmatch(file_path, pattern):
            # Found the right handler
            into = LoadInto(window.get_experiment())
            result = handler.load_file_interactive(file_path, into=into)
            if result:
                for extra_experiment in into.extra_experiments:
                    window.get_gui_experiment().add_experiment(extra_experiment)
            return result
    return False