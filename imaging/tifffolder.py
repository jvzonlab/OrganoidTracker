from imaging import Experiment
from os import path


def load_images_from_folder(experiment: Experiment, folder: str, file_name_format: str, max_frame: int = 5000):
    frame = 1
    while frame <= max_frame:
        file_name = path.join(folder, file_name_format % frame)
        if not path.isfile(file_name):
            break
        experiment.add_image(frame, file_name)
        frame += 1