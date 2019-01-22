from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent

from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.launcher import launch_window
from autotrack.gui.window import Window
from autotrack.imaging import io
from autotrack.linking_analysis import particle_flow_calculator
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
    Press F to show the detected position flow, press V to view the detected position volume"""

    def __init__(self, window: Window, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: Optional[DisplaySettings] = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)

    def _on_mouse_click(self, event: MouseEvent):
        if event.button == 1:
            position = self._get_position_at(event.xdata, event.ydata)
            if position is not None:
                self.update_status(f"Clicked on {position}")
        else:
            super()._on_mouse_click(event)

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Edit//Experiment-Merge tracking data...": self._ask_merge_experiments,
            "Edit//Experiment-Manually change data... (C)": self._show_data_editor,
            "Edit//Automatic-Cell detection...": self._show_cell_detector,
            "View//Cells-Cell divisions... (M)": self._show_mother_cells,
            "View//Cells-Track ends and cell deaths... (/deaths)": self._show_dead_cells,
            "View//Tracks-Track follower... (T)": self._show_track_follower,
            "View//Tracks-Cell fates...": self._show_cell_fates,
            "View//Tracks-Whole lineage fates...": self._show_lineage_fates,
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            self._show_track_follower()
        elif event.key == "m":
            self._show_mother_cells()
        elif event.key == "c":
            self._show_data_editor()
        elif event.key == "v":  # show volume info
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

    def _show_track_follower(self):
        from autotrack.visualizer.track_visualizer import TrackVisualizer
        track_visualizer = TrackVisualizer(self._window, self._time_point, self._z, self._display_settings)
        activate(track_visualizer)

    def _show_cell_detector(self):
        if self._experiment.get_image_stack(self._time_point) is None:
            dialog.popup_error("No images", "There are no images loaded, so we cannot detect cells.")
            return
        from autotrack.visualizer.detection_visualizer import DetectionVisualizer
        activate(DetectionVisualizer(self._window, self._time_point, self._z, self._display_settings))

    def _show_mother_cells(self):
        from autotrack.visualizer.cell_division_visualizer import CellDivisionVisualizer
        track_visualizer = CellDivisionVisualizer(self._window)
        activate(track_visualizer)

    def _show_cell_fates(self):
        from autotrack.visualizer.cell_fate_visualizer import CellFateVisualizer
        fate_visualizer = CellFateVisualizer(self._window, time_point=self._time_point, z=self._z,
                                             display_settings=self._display_settings)
        activate(fate_visualizer)

    def _show_lineage_fates(self):
        from autotrack.visualizer.lineage_fate_visualizer import LineageFateVisualizer
        fate_visualizer = LineageFateVisualizer(self._window, time_point=self._time_point, z=self._z,
                                                display_settings=self._display_settings)
        activate(fate_visualizer)

    def _show_data_editor(self):
        from autotrack.visualizer.link_and_position_editor import LinkAndPositionEditor
        editor = LinkAndPositionEditor(self._window, time_point=self._time_point, z=self._z)
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

    def _ask_merge_experiments(self):
        link_file = dialog.prompt_load_file("Select data file",
                                            [(io.FILE_EXTENSION.upper() + " files", "*." + io.FILE_EXTENSION),
                                             ("JSON files", "*.json")])
        if not link_file:
            return  # Cancelled

        new_experiment = io.load_data_file(link_file)
        self._experiment.merge(new_experiment)
        self.get_window().redraw_data()
