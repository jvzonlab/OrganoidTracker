from typing import Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent

from organoid_tracker import core
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction
from organoid_tracker.linking import nearby_position_finder
from organoid_tracker.visualizer.abstract_editor import AbstractEditor


class _InsertBeaconAction(UndoableAction):
    _beacon: Position

    def __init__(self, beacon: Position):
        self._beacon = beacon

    def do(self, experiment: Experiment) -> str:
        experiment.beacons.add(self._beacon)
        return f"Added a beacon at {self._beacon}"

    def undo(self, experiment: Experiment) -> str:
        experiment.beacons.remove(self._beacon)
        return f"Removed a beacon at {self._beacon}"


class _MoveBeaconAction(UndoableAction):
    _old_position: Position
    _new_position: Position

    def __init__(self, old_position: Position, new_position: Position):
        self._old_position = old_position
        self._new_position = new_position

    def do(self, experiment: Experiment) -> str:
        experiment.beacons.move(self._old_position, self._new_position)
        return f"Moved the beacon to {self._new_position}"

    def undo(self, experiment: Experiment) -> str:
        experiment.beacons.move(self._new_position, self._old_position)
        return f"Moved the beacon back to {self._old_position}"


class BeaconEditor(AbstractEditor):
    """Editor for beacons - abstract points that cells move around or towards. Use Insert, Shift and Delete to insert,
    shift or delete points."""

    _selected_index: Optional[int] = None
    _draw_beacon_distances: bool = False

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Positions-Toggle showing distances to positions": self._toggle_showing_beacon_distances
        }

    def _toggle_showing_beacon_distances(self):
        self._draw_beacon_distances = not self._draw_beacon_distances
        self.draw_view()

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if not self._draw_beacon_distances or dt != 0 or abs(dz) > self.MAX_Z_DISTANCE:
            return super()._on_position_draw(position, color, dz, dt)

        resolution = self._experiment.images.resolution()
        beacon = self._experiment.beacons.find_closest_beacon(position, resolution)

        if beacon is None:
            return super()._on_position_draw(position, color, dz, dt)
        is_selected = beacon.beacon_index == self._selected_index

        background_color = (1, 1, 1, 0.8) if is_selected else (0, 1, 0, 0.8)
        self._draw_annotation(position, f"{beacon.distance_um:.1f} μm", background_color=background_color)

    def _draw_extra(self):
        """Draws the selection box."""
        selected = self._selected_beacon()
        if selected is not None:
            self._draw_selection(selected, core.COLOR_CELL_CURRENT)

    def _selected_beacon(self) -> Optional[Position]:
        """Gets the currently selected beacon."""
        if self._selected_index is None:
            return None
        return self._experiment.beacons.get_beacon_by_index(self._time_point, self._selected_index)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert" or event.key == "enter":
            self._try_insert(event)
        elif event.key == "delete" or event.key == "backspace":
            self._try_remove()
        else:
            super()._on_key_press(event)

    def _try_insert(self, event: KeyEvent):
        if event.xdata is None or event.ydata is None:
            self.update_status("Place your mouse at the location where you want to insert a beacon.")
            return

        beacon = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        self._selected_index = self._experiment.beacons.get_next_index(self._time_point)
        self._perform_action(_InsertBeaconAction(beacon))

    def _try_remove(self):
        selected = self._selected_beacon()
        if selected is None:
            self.update_status("No beacon was selected - cannot delete anything.")
            return
        self._selected_index = None
        self._perform_action(ReversedAction(_InsertBeaconAction(selected)))
