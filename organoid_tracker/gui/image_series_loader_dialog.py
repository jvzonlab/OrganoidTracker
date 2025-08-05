from xml.dom.minidom import Element

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.imaging.image_file_name_pattern_finder import find_time_and_channel_pattern
import os.path

def prompt_image_series(experiment: Experiment) -> bool:
    """Prompts an image series, and loads it into the experiment. Returns whether anything was loaded."""
    full_path = dialog.prompt_load_file("Select first image file", [
        ("Single TIF or TIF series", "*.tif *.tiff"),
        ("Image per time point", "*.png *.jpg *.gif"),
        ("LIF file", "*.lif"),
        ("ND2 file", "*.nd2"),
        ("CZI file", "*.czi"),
        ("Imaris file", "*.ims")])
    if not full_path:
        return False  # Cancelled
    directory, file_name = os.path.split(full_path)

    if file_name.endswith(".lif"):
        # LIF file loading
        from organoid_tracker.image_loading import _lif, liffile_image_loader
        reader = _lif.Reader(full_path)
        series = liffile_image_loader.get_series_display_names(reader)
        series_index = option_choose_dialog.prompt_list("Choose an image serie", "Choose an image serie", "Image serie:", series)
        if series_index is not None:
            experiment.images.close_image_loader()
            liffile_image_loader.load_from_lif_reader(experiment, full_path, reader, series_index + 1)
            return True
        return False

    if file_name.endswith(".nd2"):
        # ND2 file loading
        from organoid_tracker.image_loading import nd2file_image_loader
        reader = nd2file_image_loader.Nd2File(full_path)
        max_location = reader.get_location_counts()
        location = dialog.prompt_int("Image series", f"Which image series do you want load? (1-"
                                     f"{max_location}, inclusive)", minimum=1, maximum=max_location)
        if location is not None:
            experiment.images.close_image_loader()
            nd2file_image_loader.load_image_series(experiment, reader, location)
            return True
        return False

    if file_name.endswith(".czi"):
        # CZI file loading
        from organoid_tracker.image_loading import czifile_image_loader
        reader, series_min, series_max = czifile_image_loader.read_czi_file(full_path)
        location = dialog.prompt_int("Image series", f"Which image series do you want load? ({series_min}-"
                                                     f"{series_max}, inclusive)", minimum=series_min, maximum=series_max)
        if location is not None:
            experiment.images.close_image_loader()
            czifile_image_loader.load_from_czi_reader(experiment, full_path, reader, location)
            return True
        return False

    if file_name.endswith(".ims"):
        # IMS file loading
        from organoid_tracker.image_loading import imsfile_image_loader
        experiment.images.close_image_loader()
        imsfile_image_loader.load_from_ims_file(experiment, full_path)
        return True

    file_name_pattern = find_time_and_channel_pattern(directory, file_name)
    if file_name_pattern is None:
        file_name_pattern = file_name  # Don't use a pattern if not available
        file_name_lower = file_name.lower()
        if file_name_lower.endswith(".tif") or file_name_lower.endswith(".tiff"):
            # Try as TIF container
            from organoid_tracker.image_loading import merged_tiff_image_loader
            experiment.images.close_image_loader()
            merged_tiff_image_loader.load_from_tif_file(experiment, full_path)
            return True
        dialog.popup_message("Could not read file pattern", "Could not find 't01' (or similar) in the file name \"" +
                             file_name + "\", so only one image is loaded. If you want to load a time lapse, see the"
                             " manual for supported image formats.")

    # Load and show images
    from organoid_tracker.image_loading import folder_image_loader
    experiment.images.close_image_loader()
    folder_image_loader.load_images_from_folder(experiment, directory, file_name_pattern)
    return True
