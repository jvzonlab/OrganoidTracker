from matplotlib.backend_bases import MouseEvent

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.launcher import launch_window
from organoid_tracker.imaging import io
from organoid_tracker.text_popup.position_popup import PositionPopup
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

    def _on_mouse_double_click(self, event: MouseEvent):
        if event.button == 1:
            position = self._get_position_at(event.xdata, event.ydata)
            if position is not None:
                dialog.popup_rich_text(PositionPopup(self._window, position))
        else:
            super()._on_mouse_single_click(event)

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Edit//Batch-Merge tracking data...": self._ask_merge_experiments,
            "Edit//Experiment-Manually change data... [C]": self._show_data_editor,
            "View//Analyze-Lists//Cell divisions... [M]": self._show_mother_cells,
            "View//Analyze-Lists//Cell shedding and deaths... [S]": self._show_dead_cells,
            "View//Analyze-Cell properties//Cell density...": self._show_cell_density,
            "View//Tracks-Track follower... [T]": self._show_track_follower,
            "View//Analyze-Division scores...": self._show_division_scores,
            "View//Analyze-Link scores...": self._show_link_scores,
            "View//Analyze-Analyze fate of cells//Movement arrows...": self._show_movement_arrows,
            "View//Analyze-Analyze fate of cells//Cell fates...": self._show_cell_fates,
            "View//Analyze-Analyze fate of cells//Cell compartments...": self._show_cell_compartments,
            "View//Analyze-Analyze fate of cells//Whole lineage fates...": self._show_lineage_fates,
        }

    def _get_must_show_plugin_menus(self) -> bool:
        return True

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

    def _ask_merge_experiments(self):
        link_files = dialog.prompt_load_multiple_files("Select data file", io.SUPPORTED_IMPORT_FILES)
        if len(link_files) == 0:
            return  # Cancelled

        for link_file in link_files:
            new_experiment = io.load_data_file(link_file)
            self._experiment.merge(new_experiment)
        self.get_window().redraw_data()

    def _show_division_scores(self):
        from organoid_tracker.visualizer.division_score_visualizer import DivisionScoreVisualizer
        link_score_visualizer = DivisionScoreVisualizer(self._window)
        activate(link_score_visualizer)

    def _show_link_scores(self):
        from organoid_tracker.visualizer.link_score_visualizer import LinkScoreVisualizer
        link_score_visualizer = LinkScoreVisualizer(self._window)
        activate(link_score_visualizer)