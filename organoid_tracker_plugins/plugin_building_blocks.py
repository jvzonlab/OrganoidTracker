from os import path
from typing import List, Optional, Dict, Any

from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import RegularPolygon

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui import dialog
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers, cell_fate_finder, particle_age_finder
from organoid_tracker.linking_analysis.cell_fate_finder import CellFate, CellFateType
from organoid_tracker.visualizer import Visualizer

_ROW_HEIGHT_UM = 5.5
_ROW_WIDTH = 8


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "View//Model-Unwrapped crypt honeycomb model...": lambda: _popup_lattice_model(window),
    }


class _LatticeEntry:
    position: Position
    axis_position_um: float
    angle: float
    position_type: Optional[str]
    position_fate: CellFate
    position_age: Optional[int]

    def __init__(self, position: Position, axis_position_um: float, angle: float, cell_fate: CellFate,
                 cell_type: Optional[str], cell_age: Optional[int]):
        self.position = position
        self.axis_position_um = axis_position_um
        self.angle = angle
        self.position_fate = cell_fate
        self.position_type = cell_type
        self.position_age = cell_age


class _Lattice:
    """A hexagonal lattice of all the cells at at time point."""

    _list: List[List[_LatticeEntry]]

    def __init__(self):
        self._list = []

    def generate(self, experiment: Experiment, time_point: TimePoint, required_axis_id: int):
        """Generates the lattice model for the given time point."""
        self._list.clear()
        entries = []

        for position in experiment.positions.of_time_point(time_point):
            axis_position = experiment.splines.to_position_on_original_axis(experiment.links, position)
            if axis_position is None or axis_position.axis_id != required_axis_id:
                continue
            cell_type = linking_markers.get_position_type(experiment.links, position)
            cell_fate = cell_fate_finder.get_fate(experiment, position)
            cell_age = particle_age_finder.get_age(experiment.links, position)
            resolution = experiment.images.resolution()
            pos_um = axis_position.pos * resolution.pixel_size_x_um
            angle = axis_position.calculate_angle(position, resolution)
            entries.append(_LatticeEntry(position, pos_um, angle, cell_fate, cell_type, cell_age))

        # Sort and partition
        entries.sort(key=lambda entry: entry.axis_position_um)
        for i in range(0, len(entries), _ROW_WIDTH):
            self._list.append(entries[i:i + _ROW_WIDTH])

    def get_position(self, x: float, y: float) -> Optional[Position]:
        """Gets the position plotted at the figure x and y coords."""
        y = int(round(y))
        if y < 0 or y >= len(self._list):
            return None

        row = self._list[y]
        offset_x = self._get_x_offset(y, row)
        x = int(round(x + offset_x))
        if x < 0 or x >= len(row):
            return None

        return row[x].position

    def _get_x_offset(self, y: int, row: List[_LatticeEntry]) -> float:
        offset_x = 0
        # for x, element in enumerate(row):
        #     if element.angle > 0:
        #         offset_x = x
        #         break

        if y % 2 != 0:
            offset_x -= 0.5

        return offset_x

    def plot_lattice(self, axes: Axes, gui_experiment: GuiExperiment, force_resize: bool = False):
        """Plots the lattice with one on each """
        min_x = 0
        max_x = 0

        for y, row in enumerate(self._list):
            row.sort(key=lambda value: value.angle)

            offset_x = self._get_x_offset(y, row)
            for x, element in enumerate(row):
                x -= offset_x
                min_x = min(x, min_x)
                max_x = max(x, max_x)

                position_type = gui_experiment.get_marker_by_save_name(element.position_type)
                color = self._get_color(element.position_fate, element.position_age)
                edge_color = "gray" if position_type is None else position_type.mpl_color
                edge_width = 1 if position_type is None else 2
                hex = RegularPolygon((x, y), numVertices=6, radius=0.5,
                                     orientation=0,
                                     facecolor=color, edgecolor=edge_color, linewidth=edge_width)
                axes.add_patch(hex)

        if axes.get_xlim() == (0, 1) or force_resize:
            axes.set_ylim(-1, len(self._list) + 1)
            axes.invert_yaxis()
            axes.set_xlim(min_x - 1, max_x + 1)

    def _get_color(self, position_fate: CellFate, cell_age: Optional[int]) -> MPLColor:
        """Determines the color of a cell."""
        color = (0.5, 0.5, 0.5)
        if position_fate.type == CellFateType.WILL_DIE:
            red_green_value = min(0.6, position_fate.time_points_remaining * 0.02)
            color = (1, red_green_value, red_green_value)
        elif position_fate.type == CellFateType.WILL_SHED:
            red_green_value = min(0.6, position_fate.time_points_remaining * 0.02)
            color = (red_green_value, red_green_value, 1)
        elif position_fate.type == CellFateType.WILL_DIVIDE:
            red_blue_value = min(0.8, position_fate.time_points_remaining * 0.02)
            color = (red_blue_value, 1, red_blue_value)
        elif position_fate.type == CellFateType.JUST_MOVING:
            color = (1, 1, 1)

        if cell_age is not None and cell_age < 2:
            # Add blue-ish tint to RGB color for newbord cells
            fuchsia = (1, 0, 1)  # Fuchsia color in RGB
            if cell_age == 0:
                color = fuchsia
            else:
                color = (fuchsia[0]/2 + color[0]/2, fuchsia[1]/2 + color[1]/2, fuchsia[2]/2 + color[2]/2)
        return color


class _LatticePlot(Visualizer):
    """Plots a 2D representation of the crypt. Double-click on a cell to view that cell in the images. Use Left and
    Right to move through time, Up and Down to view the other crypts in the organoid. Red = cell will die,
    green = cell will divide, white = cell is just moving, fuchsia = newborn cell, gray = unknown what happened."""

    _lattice: _Lattice
    _popup_time_point: TimePoint
    _axis_id: int

    def __init__(self, window: Window):
        super().__init__(window)

        experiment = self._experiment
        self._lattice = _Lattice()
        self._popup_time_point = TimePoint(experiment.first_time_point_number())
        self._axis_id = -1
        for axis_id, data_axis in experiment.splines.of_time_point(self._popup_time_point):
            self._axis_id = axis_id
            break

    def _on_key_press(self, event: KeyEvent):
        # Move in time
        if event.key == "left":
            try:
                self._popup_time_point = self._experiment.get_previous_time_point(self._popup_time_point)
                self.draw_view()
                self.update_status(f"Moved to time point {self._popup_time_point.time_point_number()}")
            except ValueError:
                self.update_status("There is no previous time point.")
        elif event.key == "right":
            try:
                self._popup_time_point = self._experiment.get_next_time_point(self._popup_time_point)
                self.draw_view()
                self.update_status(f"Moved to time point {self._popup_time_point.time_point_number()}")
            except ValueError:
                self.update_status("There is no next time point.")
        elif event.key == "up" or event.key == "down":
            axis_ids = []
            for axis_id, data_axis in self._experiment.splines.of_time_point(self._popup_time_point):
                axis_ids.append(axis_id)
            if len(axis_ids) == 0:
                self.update_status("No crypt axes found in time point")
                return
            try:
                current_index = axis_ids.index(self._axis_id)
            except ValueError:
                current_index = 0
            if event.key == "up":
                current_index = (current_index + 1) % len(axis_ids)
            else:
                current_index = (current_index - 1) % len(axis_ids)
            self._axis_id = axis_ids[current_index]
            self.draw_view()
            if len(axis_ids) > 1:
                self.update_status(f"Now showing data axis {self._axis_id}")
        elif event.key == "m":
            self._take_movie()
        else:
            super()._on_key_press(event)

    def draw_view(self):
        self._clear_axis()
        self._ax.set_aspect('equal')
        self._ax.get_xaxis().set_visible(False)
        self._ax.get_yaxis().set_visible(False)
        self._ax.set_title(f"Unwrapping of crypt {self._axis_id} in time point {self._popup_time_point.time_point_number()}")
        self._lattice.generate(self._experiment, self._popup_time_point, self._axis_id)
        self._lattice.plot_lattice(self._ax, self.get_window().get_gui_experiment())
        self._ax.set_ylabel("Villus-to-crypt axis")
        self._fig.canvas.draw()

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick or event.xdata is None or event.ydata is None:
            return

        # Focus on clicked cell
        position = self._lattice.get_position(event.xdata, event.ydata)
        if position is not None:
            self.get_window().get_gui_experiment().goto_position(position)
            self.update_status(f"Focused main window on {position}.")
        else:
            self.update_status("No cell found there.")

    def _take_movie(self):
        original_time_point = self._popup_time_point
        directory = dialog.prompt_directory("Choose a directory to save the images to")
        if directory is None:
            return

        # Resize figure for latest (and presumable largest) time point
        self._popup_time_point = TimePoint(self._experiment.positions.last_time_point_number())
        self._lattice.generate(self._experiment, self._popup_time_point, self._axis_id)
        self._lattice.plot_lattice(self._ax, self.get_window().get_gui_experiment(), force_resize=True)

        # Create movie time point by time point
        for time_point in self._experiment.time_points():
            file_name = path.join(directory, f"image-t{time_point.time_point_number()}.png")
            self._popup_time_point = time_point
            self.draw_view()
            self._fig.savefig(file_name)

        self._popup_time_point = original_time_point
        self.draw_view()
        self.update_status("Creation of movie complete")

def _popup_lattice_model(window: Window):
    dialog.popup_visualizer(window.get_gui_experiment(), _LatticePlot)
