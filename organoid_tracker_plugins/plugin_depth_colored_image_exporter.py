import os
from typing import Dict, Any

import matplotlib.image

from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export image//Projection-Depth-colored projection...": lambda: _export_depth_colored_image(window),
        "File//Export-Export movie//Projection-Depth-colored projection...": lambda: _export_depth_colored_movie(window)
    }


def _export_depth_colored_image(window: Window):
    from organoid_tracker.imaging import depth_colored_image_creator

    time_point = window.display_settings.time_point
    image_channel = window.display_settings.image_channel
    experiments = list(window.get_active_experiments())

    if len(experiments) == 0:
        raise UserError("No open experiments", "No experiments are open. Cannot save any image.")

    if len(experiments) == 1:
        # Case of a single experiment, prompt to save PNG file
        experiment = experiments[0]
        image_3d = experiment.images.get_image_stack(time_point, image_channel)
        if image_3d is None:
            raise UserError("No image available", "There is no image available for this time point.")

        output_file = dialog.prompt_save_file("Image location", [("PNG file", "*.png")])
        if output_file is None:
            return

        image_2d = depth_colored_image_creator.create_image(image_3d)
        matplotlib.image.imsave(output_file, image_2d)
        return

    # Multiple experiments, prompt to save a folder
    output_folder = dialog.prompt_save_file("Output location", [("Folder", "*")])
    if output_folder is None:
        return
    if os.path.isfile(output_folder):
        os.unlink(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    saved_an_image = False
    for i, experiment in enumerate(experiments):
        image_3d = experiment.images.get_image_stack(time_point, image_channel)
        if image_3d is None:
            continue

        output_file = os.path.join(output_folder, f"{i + 1}. {experiment.name}.png")
        image_2d = depth_colored_image_creator.create_image(image_3d)
        matplotlib.image.imsave(output_file, image_2d)
        saved_an_image = True
    if not saved_an_image:
        raise UserError("No images saved", "None of the open experiments had an image for the current"
                                           f" time point ({time_point.time_point_number()}) and image channel"
                                           f" ({image_channel.index_one}).")
    else:
        answer = dialog.prompt_options("Images saved", f"The images have been saved to {output_folder}.",
                                       option_1="Open that directory", option_default=DefaultOption.OK)
        if answer == 1:
            dialog.open_file(output_folder)


def _export_depth_colored_movie(window: Window):
    from organoid_tracker.imaging import depth_colored_image_creator
    import tifffile

    experiment = window.get_experiment()
    if not experiment.images.image_loader().has_images():
        raise UserError("No image available", "There is no image available for the experiment.")

    file = dialog.prompt_save_file("Image location", [("TIFF file", "*.tif")])
    if file is None:
        return

    images_copy = experiment.images.copy()
    channel = window.display_settings.image_channel

    class ImageTask(Task):
        def compute(self):
            image_movie = depth_colored_image_creator.create_movie(images_copy, channel)
            tifffile.imwrite(file, image_movie, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})
            return file

        def on_finished(self, result: Any):
            dialog.popup_message("Movie created", f"Done! The movie is now created at {result}.")

    window.get_scheduler().add_task(ImageTask())