from typing import List, Optional, Dict, Any

from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import RegularPolygon

from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.position import Position
from autotrack.gui import dialog
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers
from autotrack.visualizer import Visualizer


_ROW_HEIGHT_UM = 5.5


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "View//Model-Unwrapped crypt honeycomb model...": lambda: _popup_lattice_model(window),
    }


class _LatticeEntry:
    position: Position
    angle: float
    position_type: Optional[str]

    def __init__(self, position: Position, angle: float, cell_type: Optional[str]):
        self.position = position
        self.angle = angle
        self.position_type = cell_type


class _Lattice:
    """A hexagonal lattice of all the cells at at time point."""

    _list: List[List[_LatticeEntry]]
    _experiment: Experiment

    def __init__(self, experiment: Experiment):
        self._experiment = experiment
        self._list = []

    def generate(self, time_point: TimePoint, axis_id: int):
        """Generates the lattice model for the given time point."""
        self._list.clear()
        for position in self._experiment.positions.of_time_point(time_point):
            self._add_position(position, axis_id)

    def _add_position(self, position: Position, required_axis_id: int):
        axis_position = self._experiment.data_axes.to_position_on_original_axis(self._experiment.links, position)
        if axis_position is None or axis_position.axis_id != required_axis_id:
            return
        cell_type = linking_markers.get_position_type(self._experiment.links, position)
        resolution = self._experiment.images.resolution()
        pos_um = axis_position.pos * resolution.pixel_size_x_um
        angle = axis_position.calculate_angle(position, resolution)
        row = int(pos_um / _ROW_HEIGHT_UM)

        while len(self._list) <= row:
            self._list.append([])
        self._list[row].append(_LatticeEntry(position, angle, cell_type))

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
        for x, element in enumerate(row):
            if element.angle > 0:
                offset_x = x
                break

        if y % 2 != 0:
            offset_x -= 0.5

        return offset_x

    def plot_lattice(self, axes: Axes, gui_experiment: GuiExperiment):
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

                position_type = gui_experiment.get_position_type(element.position_type)
                color = "red" if position_type is None else position_type.mpl_color
                hex = RegularPolygon((x, y), numVertices=6, radius=0.5,
                                     orientation=0,
                                     facecolor=color, alpha=0.2, edgecolor='k')
                axes.add_patch(hex)

        if axes.get_xlim() == (0, 1):
            axes.set_ylim(-1, len(self._list) + 1)
            axes.invert_yaxis()
            axes.set_xlim(min_x - 1, max_x + 1)


class _LatticePlot(Visualizer):
    """Plots a 2D representation of the crypt. Double-click on a cell to view that cell in the images. Use Left and
    Right to move through time."""

    _lattice: _Lattice
    _time_point: TimePoint
    _axis_id: int

    def __init__(self, window: Window):
        super().__init__(window)

        experiment = self._experiment
        self._lattice = _Lattice(experiment)
        self._time_point = TimePoint(experiment.first_time_point_number())
        self._axis_id = -1
        for axis_id, data_axis in experiment.data_axes.of_time_point(self._time_point):
            self._axis_id = axis_id
            break

    def _on_key_press(self, event: KeyEvent):
        # Move in time
        if event.key == "left":
            try:
                self._time_point = self._experiment.get_previous_time_point(self._time_point)
                self.draw_view()
                self.update_status(f"Moved to time point {self._time_point.time_point_number()}")
            except ValueError:
                self.update_status("There is no previous time point.")
        elif event.key == "right":
            try:
                self._time_point = self._experiment.get_next_time_point(self._time_point)
                self.draw_view()
                self.update_status(f"Moved to time point {self._time_point.time_point_number()}")
            except ValueError:
                self.update_status("There is no next time point.")
        elif event.key == "up" or event.key == "down":
            axis_ids = []
            for axis_id, data_axis in self._experiment.data_axes.of_time_point(self._time_point):
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
        else:
            super()._on_key_press(event)

    def draw_view(self):
        self._clear_axis()
        self._ax.set_aspect('equal')
        self._ax.set_title(f"Unwrapped crypt of crypt {self._axis_id} in time point {self._time_point.time_point_number()}")
        self._lattice.generate(self._time_point, self._axis_id)
        self._lattice.plot_lattice(self._ax, self.get_window().get_gui_experiment())
        self._fig.canvas.draw()

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return

        # Focus on clicked cell
        position = self._lattice.get_position(event.xdata, event.ydata)
        if position is not None:
            self.get_window().get_gui_experiment().goto_position(position)
            self.update_status(f"Focused main window on {position}.")
        else:
            self.update_status("No cell found there.")

def _popup_lattice_model(window: Window):
    dialog.popup_visualizer(window.get_gui_experiment(), _LatticePlot)
