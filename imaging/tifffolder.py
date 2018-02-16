from imaging import Experiment
from os import path
from numpy import ndarray
import tifffile


def load_images_from_folder(experiment: Experiment, folder: str, file_name_format: str, min_frame: int = 0,
                            max_frame: int = 5000):
    frame = max(1, min_frame)
    while frame <= max_frame:
        file_name = path.join(folder, file_name_format % frame)

        if not path.isfile(file_name):
            break

        experiment.add_image_loader(frame, _create_image_loader(file_name))
        frame += 1


def _create_image_loader(file_name: str):
    def image_loader() -> ndarray:
        with tifffile.TiffFile(file_name) as f:
            # noinspection PyTypeChecker
            return f.asarray(maxworkers=None)  # maxworkers=None makes image loader work on half of all cores
    return image_loader
