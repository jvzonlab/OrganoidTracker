from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent

from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.launcher import launch_window
from autotrack.gui.window import Window
from autotrack.imaging import io
from autotrack.linking import particle_flow
from autotrack.manual_tracking import guizela_data_importer
from autotrack.visualizer import activate, DisplaySettings
from autotrack.visualizer.abstract_image_visualizer import AbstractImageVisualizer


def show(experiment: Experiment):
    """Creates a standard visualizer for an experiment."""
    window = launch_window(experiment)
    visualizer = StandardImageVisualizer(window)
    activate(visualizer)


class StandardImageVisualizer(AbstractImageVisualizer):
    """Cell and image viewer

    Moving: left/right moves in time, up/down in the z-direction and type '/t30' + ENTER to jump to time point 30
    Press F to show the detected particle flow, press V to view the detected particle volume"""

    def __init__(self, window: Window, time_point_number: Optional[int] = None, z: int = 14,
                 display_settings: Optional[DisplaySettings] = None):
        super().__init__(window, time_point_number=time_point_number, z=z, display_settings=display_settings)

    def _on_mouse_click(self, event: MouseEvent):
        if event.dblclick and event.button == 1:
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is not None:
                self.__display_cell_division_scores(particle)
        else:
            super()._on_mouse_click(event)

    def __display_cell_division_scores(self, particle):
        cell_divisions = list(self._experiment.scores.of_mother(particle))
        cell_divisions.sort(key=lambda d: d.score.total(), reverse=True)
        displayed_items = 0
        text = ""
        for scored_family in cell_divisions:
            if displayed_items >= 2:
                text += "... and " + str(len(cell_divisions) - displayed_items) + " more"
                break
            text += str(displayed_items + 1) + ". " + str(scored_family.family) + ", score: " \
                    + str(scored_family.score).replace(",", ",\n\t") + "\n"
            displayed_items += 1
        if text:
            self.update_status("Possible cell division scores:\n" + text)
        else:
            self.update_status("No cell division scores found")

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Edit/Add-Add positions and shapes...": lambda: self._ask_add_positions_from_file,
            "Edit/Add-Add links, scores and warnings...": lambda: self._ask_add_links_from_file,
            "Edit/Add-Add positions and links from Guizela's format...": lambda: self._ask_add_guizela_tracks,
            "Edit/Manual-Manually change data... (C)": self._show_data_editor,
            "Edit/Automatic-Cell detection...": self._show_cell_detector,
            "View/Cell-Cell divisions (M)": self._show_mother_cells,
            "View/Cell-Track ends and cell deaths (/deaths)": self._show_dead_cells
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            from autotrack.visualizer.track_visualizer import TrackVisualizer
            track_visualizer = TrackVisualizer(self._window, self._time_point.time_point_number(), self._z,
                                               self._display_settings)
            activate(track_visualizer)
        elif event.key == "m":
            self._show_mother_cells()
        elif event.key == "c":
            self._show_data_editor()
        elif event.key == "v":  # show volume info
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is None:
                self.update_status("No particle at mouse position")
                return
            shape = self._experiment.particles.get_shape(particle)
            try:
                self.update_status(f"Volume of {particle} is {shape.volume():.2f} px3")
            except NotImplementedError:
                self.update_status(f"The {particle} has no volume information stored")
        elif event.key == "f":  # show flow info
            particle = self._get_particle_at(event.xdata, event.ydata)
            particles_of_time_point = self._experiment.particles.of_time_point(self._time_point)
            links = self._experiment.links.graph
            if particle is not None and links is not None:
                self.update_status("Flow toward previous frame: " +
                                   str(particle_flow.get_flow_to_previous(links, particles_of_time_point, particle)) +
                                   "\nFlow towards next frame: " +
                                   str(particle_flow.get_flow_to_next(links, particles_of_time_point, particle)))
        else:
            super()._on_key_press(event)

    def _show_cell_detector(self):
        if self._experiment.get_image_stack(self._time_point) is None:
            dialog.popup_error("No images", "There are no images loaded, so we cannot detect cells.")
            return
        from autotrack.visualizer.detection_visualizer import DetectionVisualizer
        activate(DetectionVisualizer(self._window, self._time_point.time_point_number(), self._z,
                                     self._display_settings))

    def _show_mother_cells(self):
        from autotrack.visualizer.cell_division_visualizer import CellDivisionVisualizer
        track_visualizer = CellDivisionVisualizer(self._window)
        activate(track_visualizer)

    def _show_data_editor(self):
        from autotrack.visualizer.link_and_position_editor import LinkAndPositionEditor
        editor = LinkAndPositionEditor(self._window, time_point_number=self._time_point.time_point_number(),
                                       z=self._z)
        activate(editor)

    def _on_command(self, command: str) -> bool:
        if command == "deaths":
            self._show_dead_cells()
            return True
        if command == "divisions":
            self._show_mother_cells()
            return True
        if command == "help":
            self.update_status("Available commands:\n"
                               "/deaths - views cell deaths.\n"
                               "/divisions - views cell divisions.\n"
                               "/t20 - jumps to time point 20 (also works for other time points")
            return True
        return super()._on_command(command)

    def _show_dead_cells(self):
        from autotrack.visualizer.cell_death_visualizer import CellTrackEndVisualizer
        activate(CellTrackEndVisualizer(self._window, None))

    def _ask_add_positions_from_file(self):
        cell_file = dialog.prompt_load_file("Select positions file", [("JSON file", "*.json")])
        if not cell_file:
            return  # Cancelled

        io.load_positions_and_shapes_from_json(self._experiment, cell_file)
        self.draw_view()

    def _ask_add_links_from_file(self):
        link_file = dialog.prompt_load_file("Select link file", [("JSON file", "*.json")])
        if not link_file:
            return  # Cancelled

        io.load_linking_result(self._experiment, str(link_file))
        self.draw_view()

    def _ask_add_guizela_tracks(self):
        """Loads the tracks in Guizela's format."""
        folder = dialog.prompt_directory("Open folder with Guizela's tracks...")
        if not folder:
            return  # Cancelled

        guizela_data_importer.add_data_to_experiment(self._experiment, folder)
        self.draw_view()
