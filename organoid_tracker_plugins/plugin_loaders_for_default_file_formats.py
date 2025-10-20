import os
from typing import List, Set

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.gui import option_choose_dialog, dialog
from organoid_tracker.imaging import image_file_name_pattern_finder, io
from organoid_tracker.imaging.file_loader import FileLoader, FileLoaderType


class _TifFileLoader(FileLoader):
    def get_name(self) -> str:
        return "Single TIF or TIF series"

    def get_file_patterns(self) -> Set[str]:
        return {"*.tif", "*.tiff"}

    def load_file_interactive(self, file_path: str, *, into: Experiment) -> bool:
        from organoid_tracker.image_loading import folder_image_loader
        into.images.close_image_loader()

        directory, file_name = os.path.split(file_path)
        file_name_pattern = image_file_name_pattern_finder.find_time_and_channel_pattern(directory, file_name)
        if file_name_pattern is None:
            # No pattern found, assume it's a merged TIFF file
            from organoid_tracker.image_loading import merged_tiff_image_loader
            into.images.close_image_loader()
            merged_tiff_image_loader.load_from_tif_file(into, file_path)
            return True

        folder_image_loader.load_images_from_folder(into, directory, file_name_pattern)
        return True

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _LifFileLoader(FileLoader):
    def get_name(self) -> str:
        return "LIF file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.lif"}

    def load_file_interactive(self, file_path: str, *, into: Experiment) -> bool:
        from organoid_tracker.image_loading import _lif, liffile_image_loader
        reader = _lif.Reader(file_path)
        series = liffile_image_loader.get_series_display_names(reader)
        series_index = option_choose_dialog.prompt_list("Choose an image serie", "Choose an image serie",
                                                        "Image serie:", series)
        if series_index is not None:
            into.images.close_image_loader()
            liffile_image_loader.load_from_lif_reader(into, file_path, reader, series_index + 1)
            return True
        return False

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _Nd2FileLoader(FileLoader):
    def get_name(self) -> str:
        return "ND2 file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.nd2"}

    def load_file_interactive(self, file_path: str, *, into: Experiment) -> bool:
        from organoid_tracker.image_loading import nd2file_image_loader
        reader = nd2file_image_loader.Nd2File(file_path)
        max_location = reader.get_location_counts()
        location = dialog.prompt_int("Image series", f"Which image series do you want load? (1-"
                                     f"{max_location}, inclusive)", minimum=1, maximum=max_location)
        if location is not None:
            into.images.close_image_loader()
            nd2file_image_loader.load_image_series(into, reader, location)
            return True
        return False

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _CziFileLoader(FileLoader):
    def get_name(self) -> str:
        return "CZI file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.czi"}

    def load_file_interactive(self, file_path: str, *, into: Experiment) -> bool:
        from organoid_tracker.image_loading import czifile_image_loader
        reader, series_min, series_max = czifile_image_loader.read_czi_file(file_path)
        location = dialog.prompt_int("Image series", f"Which image series do you want load? ({series_min}-"
                                                     f"{series_max}, inclusive)", minimum=series_min, maximum=series_max)
        if location is not None:
            into.images.close_image_loader()
            czifile_image_loader.load_from_czi_reader(into, file_path, reader, location)
            return True
        return False

    def get_type(self) -> FileLoaderType:
        return FileLoaderType.IMAGE


class _ImsFileLoader(FileLoader):
    def get_name(self) -> str:
        return "Imaris file"

    def get_file_patterns(self) -> Set[str]:
        return {"*.ims"}

    def load_file_interactive(self, file_path: str, *, into: Experiment) -> bool:
        from organoid_tracker.image_loading import imsfile_image_loader
        into.images.close_image_loader()
        imsfile_image_loader.load_from_ims_file(into, file_path)
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

    def load_file_interactive(self, file_path: str, *, into: Experiment) -> bool:
        into.clear_tracking_data()
        io.load_data_file(file_path, experiment=into)
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
            _TrackingFileLoader(io.FILE_EXTENSION.upper() + " file", {"*." + io.FILE_EXTENSION}),
            _TrackingFileLoader("Old detection or linking files", {"*.json"}),
            _TrackingFileLoader("Cell tracking challenge files", {"*.txt"}),
            _TrackingFileLoader("TrackMate file", {"*.xml"}),
            _TrackingFileLoader("GEFF file", {"*.geff*", ".zgroup"}),
            _TrackingFileLoader("Guizela's tracking files", {"track_00000.p"})]
