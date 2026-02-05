import os
from typing import List, Set, Optional

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import option_choose_dialog
from organoid_tracker.imaging import image_file_name_pattern_finder, io
from organoid_tracker.imaging.file_loader import FileLoader, FileLoaderType, LoadInto


def _prompt_series(series_names: List[str], allow_all: bool) -> List[int]:
    """Prompts the user to select an image series from the given list.

    If allow_all is True, the first option will be "« All series »".

    Returns the indices of the selected series (beware, 1-based) as a list of one element. If the user cancels, returns
    an empty list. If the user selects "All series", returns a list of the length of series_names, containing all
    indices.
    """
    if len(series_names) == 1:
        return [1]  # Only one series, no need to prompt

    if allow_all:
        series_names.insert(0, "« All series »")
    selection = option_choose_dialog.prompt_list("Choose an image serie", "Choose an image serie",
                                                "Image serie:", series_names)
    if selection is None:
        return []  # User pressed cancel
    if allow_all:
        # Adjust for "All series" option
        if selection == 0:
            return list(range(1, len(series_names)))  # All series (1-based, excluding "All series" option)
        return [selection]  # Already 1-based, since option 0 is "All series"
    return [selection + 1]  # Convert to 1-based index


class _TifFileLoader(FileLoader):
    def get_name(self) -> str:
        return "Single TIF or TIF series"

    def get_file_patterns(self) -> Set[str]:
        return {"*.tif", "*.tiff"}

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        from organoid_tracker.image_loading import folder_image_loader
        into.experiment.images.close_image_loader()

        directory, file_name = os.path.split(file_path)
        file_name_pattern = image_file_name_pattern_finder.find_time_and_channel_pattern(directory, file_name)
        if file_name_pattern is None:
            # No pattern found, assume it's a merged TIFF file
            from organoid_tracker.image_loading import merged_tiff_image_loader
            into.experiment.images.close_image_loader()
            merged_tiff_image_loader.load_from_tif_file(into.experiment, file_path)
            return True

        folder_image_loader.load_images_from_folder(into.experiment, directory, file_name_pattern)
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _LifFileLoader(FileLoader):
    def get_name(self) -> str:
        return "LIF file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.lif"}

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        from organoid_tracker.image_loading import _lif, liffile_image_loader
        reader = _lif.Reader(file_path)
        series = liffile_image_loader.get_series_display_names(reader)
        series_indices = _prompt_series(series, into.allow_extra_tabs)
        if len(series_indices) == 0:
            return False  # User pressed cancel

        # Close any existing image loader
        into.experiment.images.close_image_loader()

        # Load the new experiments
        reader_used = False
        for series_index in series_indices:
            if reader_used:
                # Prevent multiple tabs from using the same reader instance
                reader = _lif.Reader(file_path)

            new_experiment = into.next_experiment()
            liffile_image_loader.load_from_lif_reader(new_experiment, file_path, reader, series_index)
            reader_used = True
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _Nd2FileLoader(FileLoader):
    def get_name(self) -> str:
        return "ND2 file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.nd2"}

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        from organoid_tracker.image_loading import nd2file_image_loader
        reader = nd2file_image_loader.Nd2File(file_path)
        max_location = reader.get_location_counts()

        name_list = [f"Series {i+1}" for i in range(max_location)]
        series_indices = _prompt_series(name_list, into.allow_extra_tabs)
        if len(series_indices) == 0:
            return False  # User pressed cancel

        # Close any existing image loader
        into.experiment.images.close_image_loader()

        # Load the new experiments
        reader_used = False
        for series_index in series_indices:
            if reader_used:
                # Prevent multiple tabs from using the same reader instance
                reader = nd2file_image_loader.Nd2File(file_path)
            new_experiment = into.next_experiment()
            nd2file_image_loader.load_image_series(new_experiment, reader, series_index)
            reader_used = True
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _CziFileLoader(FileLoader):
    def get_name(self) -> str:
        return "CZI file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.czi"}

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        from organoid_tracker.image_loading import czifile_image_loader
        reader, series_min, series_max = czifile_image_loader.read_czi_file(file_path)

        series_names = [f"Series {i}" for i in range(series_min, series_max + 1)]
        locations = _prompt_series(series_names, into.allow_extra_tabs)
        if len(locations) == 0:
            return False  # User pressed cancel

        # Close any existing image loader
        into.experiment.images.close_image_loader()

        # Load the new experiments
        reader_used = False
        for location in locations:
            if reader_used:
                # Prevent multiple tabs from using the same reader instance
                reader, _, _ = czifile_image_loader.read_czi_file(file_path)
            series_index = series_min + location - 1
            czifile_image_loader.load_from_czi_reader(into.next_experiment(), file_path, reader, series_index)
            reader_used = True
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _ImsFileLoader(FileLoader):
    def get_name(self) -> str:
        return "Imaris file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.ims"}

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        from organoid_tracker.image_loading import imsfile_image_loader
        into.experiment.images.close_image_loader()
        imsfile_image_loader.load_from_ims_file(into.experiment, file_path)
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _ZarrFileLoader(FileLoader):
    def get_name(self) -> str:
        return "Zarr file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.zarr*", "*.zgroup"}

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        from organoid_tracker.image_loading import zarr_image_loader
        into.experiment.images.close_image_loader()
        zarr_image_loader.load_from_zarr_file(into.experiment, file_path)
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _TrackingFileLoader(FileLoader):
    """For loading a file supported by io.load_data_file()."""

    _name: str
    _file_extensions: Set[str]

    def __init__(self, name: str, file_patterns: Set[str]):
        self._name = name
        self._file_extensions = file_patterns

    def get_name(self) -> str:
        return self._name

    def get_file_patterns(self) -> Set[str]:
        return self._file_extensions

    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        into.experiment.clear_tracking_data()
        io.load_data_file(file_path, experiment=into.experiment)
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.TRACKING


def get_file_loaders() -> List[FileLoader]:
    """Returns the default file formats that are supported by the program."""
    return [_TifFileLoader(),
            _LifFileLoader(),
            _Nd2FileLoader(),
            _CziFileLoader(),
            _ImsFileLoader(),
            _ZarrFileLoader(),
            _TrackingFileLoader(io.FILE_EXTENSION.upper() + " file", {"*." + io.FILE_EXTENSION}),
            _TrackingFileLoader("Old detection or linking files", {"*.json"}),
            _TrackingFileLoader("Cell tracking challenge files", {"*.txt"}),
            _TrackingFileLoader("TrackMate file", {"*.xml"}),
            _TrackingFileLoader("GEFF file", {"*.geff*", ".zgroup"}),
            _TrackingFileLoader("Guizela's tracking files", {"track_00000.p"})]
