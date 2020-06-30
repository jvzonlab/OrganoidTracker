from typing import Any

from matplotlib import pyplot
from matplotlib.backend_bases import KeyEvent, MouseEvent
from tifffile import tifffile

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.launcher import launch_window
from organoid_tracker.gui.threading import Task
from organoid_tracker.imaging import io
from organoid_tracker.linking_analysis import particle_flow_calculator
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_image_visualizer import AbstractImageVisualizer


def show(experiment: Experiment):
    """Creates a standard visualizer for an experiment."""
    window = launch_window(experiment)
    visualizer = StandardImageVisualizer(window)
    activate(visualizer)


class StandardImageVisualizer(AbstractImageVisualizer):
    """Cell and image viewer

    Moving: left/right moves in time, up/down or scroll in the z-direction, type '/t30' + ENTER to jump to time
    point 30 and type '/z10' + ENTER to jump to z-layer 10."""

    def _on_mouse_click(self, event: MouseEvent):
        if event.button == 1:
            position = self._get_position_at(event.xdata, event.ydata)
            if position is not None:
                data = dict(self._experiment.position_data.find_all_data_of_position(position))

                scores = list(self._experiment.scores.of_mother(position))
                scores.sort(key=lambda scored_family: scored_family.score.total(), reverse=True)
                score_str = ""
                for score in scores:
                    score_str += f"\nDivision score: {score.score.total()}"

                self.update_status(f"Clicked on {position}.\n  Data: {data} {score_str}")
        else:
            super()._on_mouse_click(event)

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "File//Export-Export 2D depth-colored image//This time point...": self._export_depth_colored_image,
            "File//Export-Export 2D depth-colored image//All time points...": self._export_depth_colored_movie,
            "Edit//Experiment-Merge tracking data...": self._ask_merge_experiments,
            "Edit//Experiment-Manually change data... [C]": self._show_data_editor,
            "View//Cells-Cell divisions... [M]": self._show_mother_cells,
            "View//Cells-Cell shedding and deaths... [S]": self._show_dead_cells,
            "View//Cells-Cell density...": self._show_cell_density,
            "View//Cells-Cell curvature...": self._show_cell_curvature,
            "View//Tracks-Track follower... [T]": self._show_track_follower,
            "View//Tracks-Movement arrows...": self._show_movement_arrows,
            "View//Tracks-Cell fates...": self._show_cell_fates,
            "View//Tracks-Cell compartments...": self._show_cell_compartments,
            "View//Tracks-Whole lineage fates...": self._show_lineage_fates,
        }

    def _get_must_show_plugin_menus(self) -> bool:
        return True

    def _export_depth_colored_image(self):
        image_3d = self._experiment.images.get_image_stack(self._time_point, self._display_settings.image_channel)
        if image_3d is None:
            raise UserError("No image available", "There is no image available for this time point.")

        file = dialog.prompt_save_file("Image location", [("PNG file", "*.png")])
        if file is None:
            return

        from organoid_tracker.imaging import depth_colored_image_creator
        image_2d = depth_colored_image_creator.create_image(image_3d)
        pyplot.imsave(file, image_2d)

    def _export_depth_colored_movie(self):
        if not self._experiment.images.image_loader().has_images():
            raise UserError("No image available", "There is no image available for the experiment.")

        file = dialog.prompt_save_file("Image location", [("TIFF file", "*.tif")])
        if file is None:
            return

        images_copy = self._experiment.images.copy()
        channel = self._display_settings.image_channel

        class ImageTask(Task):
            def compute(self):
                from organoid_tracker.imaging import depth_colored_image_creator
                image_movie = depth_colored_image_creator.create_movie(images_copy, channel)
                tifffile.imsave(file, image_movie, compress=6)
                return file

            def on_finished(self, result: Any):
                dialog.popup_message("Movie created", f"Done! The movie is now created at {result}.")

        self._window.get_scheduler().add_task(ImageTask())

    def _show_track_follower(self):
        from organoid_tracker.visualizer.track_visualizer import TrackVisualizer
        track_visualizer = TrackVisualizer(self._window)
        activate(track_visualizer)

    def _show_movement_arrows(self):
        from organoid_tracker.visualizer.cell_movement_visualizer import CellMovementVisualizer
        movement_visualizer = CellMovementVisualizer(self._window)
        activate(movement_visualizer)

    def _show_mother_cells(self):
        from organoid_tracker.visualizer.cell_division_visualizer import CellDivisionVisualizer
        track_visualizer = CellDivisionVisualizer(self._window)
        activate(track_visualizer)

    def _show_cell_fates(self):
        from organoid_tracker.visualizer.cell_fate_visualizer import CellFateVisualizer
        fate_visualizer = CellFateVisualizer(self._window)
        activate(fate_visualizer)

    def _show_cell_compartments(self):
        from organoid_tracker.visualizer.cell_compartment_visualizer import CellCompartmentVisualizer
        compartment_visualizer = CellCompartmentVisualizer(self._window)
        activate(compartment_visualizer)

    def _show_lineage_fates(self):
        from organoid_tracker.visualizer.lineage_fate_visualizer import LineageFateVisualizer
        fate_visualizer = LineageFateVisualizer(self._window)
        activate(fate_visualizer)

    def _show_data_editor(self):
        from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor
        editor = LinkAndPositionEditor(self._window)
        activate(editor)

    def _on_command(self, command: str) -> bool:
        if command == "help":
            self.update_status("Available commands:\n"
                               "/deaths - views cell deaths.\n"
                               "/divisions - views cell divisions.\n"
                               "/t20 - jumps to time point 20 (also works for other time points")
            return True
        return super()._on_command(command)

    def _show_dead_cells(self):
        from organoid_tracker.visualizer.cell_death_visualizer import CellTrackEndVisualizer
        activate(CellTrackEndVisualizer(self._window, None))

    def _show_cell_density(self):
        from organoid_tracker.visualizer.cell_density_visualizer import CellDensityVisualizer
        activate(CellDensityVisualizer(self._window))

    def _show_cell_curvature(self):
        from organoid_tracker.visualizer.cell_curvature_visualizer import CellCurvatureVisualizer
        activate(CellCurvatureVisualizer(self._window))

    def _ask_merge_experiments(self):
        link_files = dialog.prompt_load_multiple_files("Select data file", io.SUPPORTED_IMPORT_FILES)
        if len(link_files) == 0:
            return  # Cancelled

        for link_file in link_files:
            new_experiment = io.load_data_file(link_file)
            self._experiment.merge(new_experiment)
        self.get_window().redraw_data()
