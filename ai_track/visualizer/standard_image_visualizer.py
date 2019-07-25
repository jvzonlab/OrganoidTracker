from typing import Optional, Any

from matplotlib import pyplot
from matplotlib.backend_bases import KeyEvent, MouseEvent
from tifffile import tifffile

from ai_track.core import TimePoint, UserError
from ai_track.core.experiment import Experiment
from ai_track.gui import dialog
from ai_track.gui.launcher import launch_window
from ai_track.gui.threading import Task
from ai_track.gui.window import Window
from ai_track.imaging import io
from ai_track.linking_analysis import particle_flow_calculator
from ai_track.visualizer import activate, DisplaySettings
from ai_track.visualizer.abstract_image_visualizer import AbstractImageVisualizer


def show(experiment: Experiment):
    """Creates a standard visualizer for an experiment."""
    window = launch_window(experiment)
    visualizer = StandardImageVisualizer(window)
    activate(visualizer)


class StandardImageVisualizer(AbstractImageVisualizer):
    """Cell and image viewer

    Moving: left/right moves in time, up/down in the z-direction and type '/t30' + ENTER to jump to time point 30
    Press F to show the detected position flow, press V to view the detected position volume"""

    def __init__(self, window: Window, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: Optional[DisplaySettings] = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)

    def _on_mouse_click(self, event: MouseEvent):
        if event.button == 1:
            position = self._get_position_at(event.xdata, event.ydata)
            if position is not None:
                data = dict(self._experiment.links.find_all_data_of_position(position))
                shape = self._experiment.positions.get_shape(position)
                self.update_status(f"Clicked on {position}.\n  Data: {data}\n  Shape: {shape}")
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

    def _on_key_press(self, event: KeyEvent):
        if event.key == "v":  # show volume info
            position = self._get_position_at(event.xdata, event.ydata)
            if position is None:
                self.update_status("No position at mouse position")
                return
            shape = self._experiment.positions.get_shape(position)
            try:
                self.update_status(f"Volume of {position} is {shape.volume():.2f} px3")
            except NotImplementedError:
                self.update_status(f"The {position} has no volume information stored")
        elif event.key == "f":  # show flow info
            position = self._get_position_at(event.xdata, event.ydata)
            positions_of_time_point = self._experiment.positions.of_time_point(self._time_point)
            links = self._experiment.links
            if position is not None and links.has_links():
                self.update_status("Flow toward previous frame: " +
                                   str(particle_flow_calculator.get_flow_to_previous(links, positions_of_time_point, position)) +
                                   "\nFlow towards next frame: " +
                                   str(particle_flow_calculator.get_flow_to_next(links, positions_of_time_point, position)))
        else:
            super()._on_key_press(event)

    def _export_depth_colored_image(self):
        image_3d = self._experiment.images.get_image_stack(self._time_point, self._display_settings.image_channel)
        if image_3d is None:
            raise UserError("No image available", "There is no image available for this time point.")

        file = dialog.prompt_save_file("Image location", [("PNG file", "*.png")])
        if file is None:
            return

        from ai_track.imaging import depth_colored_image_creator
        image_2d = depth_colored_image_creator.create_image(image_3d)
        pyplot.imsave(file, image_2d)

    def _export_depth_colored_movie(self):
        if not self._experiment.images.image_loader().has_images():
            raise UserError("No image available", "There is no image available for the experiment.")

        file = dialog.prompt_save_file("Image location", [("TIFF file", "*.tif")])
        if file is None:
            return

        images_copy = self._experiment.images.copy()

        class ImageTask(Task):
            def compute(self):
                from ai_track.imaging import depth_colored_image_creator
                image_movie = depth_colored_image_creator.create_movie(images_copy)
                tifffile.imsave(file, image_movie, compress=6)
                return file

            def on_finished(self, result: Any):
                dialog.popup_message("Movie created", f"Done! The movie is now created at {result}.")

        self._window.get_scheduler().add_task(ImageTask())

    def _show_track_follower(self):
        from ai_track.visualizer.track_visualizer import TrackVisualizer
        track_visualizer = TrackVisualizer(self._window, self._time_point, self._z, self._display_settings)
        activate(track_visualizer)

    def _show_movement_arrows(self):
        from ai_track.visualizer.cell_movement_visualizer import CellMovementVisualizer
        movement_visualizer = CellMovementVisualizer(self._window, time_point=self._time_point, z=self._z,
                                                     display_settings=self._display_settings)
        activate(movement_visualizer)

    def _show_mother_cells(self):
        from ai_track.visualizer.cell_division_visualizer import CellDivisionVisualizer
        track_visualizer = CellDivisionVisualizer(self._window)
        activate(track_visualizer)

    def _show_cell_fates(self):
        from ai_track.visualizer.cell_fate_visualizer import CellFateVisualizer
        fate_visualizer = CellFateVisualizer(self._window, time_point=self._time_point, z=self._z,
                                             display_settings=self._display_settings)
        activate(fate_visualizer)

    def _show_cell_compartments(self):
        from ai_track.visualizer.cell_compartment_visualizer import CellCompartmentVisualizer
        compartment_visualizer = CellCompartmentVisualizer(self._window, time_point=self._time_point, z=self._z,
                                                           display_settings=self._display_settings)
        activate(compartment_visualizer)

    def _show_lineage_fates(self):
        from ai_track.visualizer.lineage_fate_visualizer import LineageFateVisualizer
        fate_visualizer = LineageFateVisualizer(self._window, time_point=self._time_point, z=self._z,
                                                display_settings=self._display_settings)
        activate(fate_visualizer)

    def _show_data_editor(self):
        from ai_track.visualizer.link_and_position_editor import LinkAndPositionEditor
        editor = LinkAndPositionEditor(self._window, time_point=self._time_point, z=self._z)
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
        from ai_track.visualizer.cell_death_visualizer import CellTrackEndVisualizer
        activate(CellTrackEndVisualizer(self._window, None))

    def _show_cell_density(self):
        from ai_track.visualizer.cell_density_visualizer import CellDensityVisualizer
        activate(CellDensityVisualizer(self._window, time_point=self._time_point, z=self._z,
                                       display_settings=self._display_settings))

    def _show_cell_curvature(self):
        from ai_track.visualizer.cell_curvature_visualizer import CellCurvatureVisualizer
        activate(CellCurvatureVisualizer(self._window, time_point=self._time_point, z=self._z,
                                       display_settings=self._display_settings))

    def _ask_merge_experiments(self):
        link_files = dialog.prompt_load_multiple_files("Select data file", [
            (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION),
            ("Detection or linking files", "*.json"),
            ("Guizela's tracking files", "lineages.p")])
        if len(link_files) == 0:
            return  # Cancelled

        for link_file in link_files:
            new_experiment = io.load_data_file(link_file)
            self._experiment.merge(new_experiment)
        self.get_window().redraw_data()
