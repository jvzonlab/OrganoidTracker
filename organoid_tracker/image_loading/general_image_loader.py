"""Tries to find out which image loader to use based on two settings, container and patter
n."""
import os.path
from typing import Dict, Any

from organoid_tracker.core.experiment import Experiment


def load_images(experiment: Experiment, container: str, pattern: str,
                min_time_point: int = 0, max_time_point: int = 1000000000):
    """Loads images from any of the supported formats. The container is a file or a directory, pattern is the format
    used to search within that file or directory. For a sequence of TIFF files, container will be a directroy and
    pattern the pattern of files in that directory. For a LIF file, container will be the LIF file, and pattern the name
    of the experiment within that file. Etc."""
    if container.endswith(".lif"):  # Try as LIF file
        from organoid_tracker.image_loading import liffile_image_loader
        liffile_image_loader.load_from_lif_file(experiment, container, pattern, min_time_point, max_time_point)
        return
    if container.endswith(".nd2"):
        from organoid_tracker.image_loading import nd2file_image_loader
        nd2file_image_loader.load_image_series_from_config(experiment, container, pattern, min_time_point, max_time_point)
        return
    if container.endswith(".tif"):
        from organoid_tracker.image_loading import merged_tiff_image_loader
        merged_tiff_image_loader.load_from_tif_file(experiment, container, min_time_point, max_time_point)
        return
    if not os.path.exists(container):
        raise ValueError("File or directory does not exist: " + container)
    if os.path.isdir(container):  # Try as images folder
        from organoid_tracker.image_loading import folder_image_loader
        folder_image_loader.load_images_from_folder(experiment, container, pattern, min_time_point, max_time_point)
        return
    raise ValueError("Unknown file format: " + container + " " + pattern)


def load_images_from_dictionary(experiment: Experiment, dictionary: Dict[str, Any],
                                min_time_point: int = 0, max_time_point: int = 1000000000):
    """For loading images that have been stored using image_loader.serialize_to_dictionary()"""
    if "images_channel_summing" in dictionary:
        # Need to load the original image loader
        load_images_from_dictionary(experiment, dictionary["images_original"], min_time_point, max_time_point)
        all_channels = experiment.images.get_channels()

        # Sum the appropriate channels
        summed_channels = list()
        for channel_list in dictionary["images_channel_summing"]:
            summed_channels.append([all_channels[index_one - 1] for index_one in channel_list])
        from organoid_tracker.image_loading import builtin_merging_image_loaders
        experiment.images.image_loader(builtin_merging_image_loaders.ChannelSummingImageLoader(
            experiment.images.image_loader(), summed_channels))
        return

    if "images_channel_appending" in dictionary:
        image_loaders = list()
        for image_dict in dictionary["images_channel_appending"]:
            load_images_from_dictionary(experiment, image_dict, min_time_point, max_time_point)
            image_loaders.append(experiment.images.image_loader())

        from organoid_tracker.image_loading import builtin_merging_image_loaders
        experiment.images.image_loader(builtin_merging_image_loaders.ChannelAppendingImageLoader(image_loaders))
        return

    if "images_time_appending" in dictionary:
        image_loaders = list()
        for image_dict in dictionary["images_time_appending"]:
            load_images_from_dictionary(experiment, image_dict, min_time_point, max_time_point)
            image_loaders.append(experiment.images.image_loader())

        from organoid_tracker.image_loading import builtin_merging_image_loaders
        experiment.images.image_loader(builtin_merging_image_loaders.TimeAppendingImageLoader(image_loaders))
        return

    if "images_container" in dictionary and "images_pattern" in dictionary:
        # Use the simpler method of loading images
        load_images(experiment, dictionary["images_container"], dictionary["images_pattern"], min_time_point,
                    max_time_point)
        return

    raise ValueError("Unknown file format: " + str(dictionary))
