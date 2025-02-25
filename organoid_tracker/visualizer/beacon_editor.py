from functools import partial
from typing import Optional, Dict, Any, List

from matplotlib.backend_bases import KeyEvent, MouseEvent

from organoid_tracker import core
from organoid_tracker.core.beacon_collection import Beacon
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.position import Position
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer.abstract_editor import AbstractEditor
from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor


class _InsertBeaconAction(UndoableAction):
    _beacon: Beacon

    def __init__(self, beacon: Beacon):
        self._beacon = beacon

    def do(self, experiment: Experiment) -> str:
        experiment.beacons.add(self._beacon.position, self._beacon.beacon_type)
        if self._beacon.beacon_type is not None:
            return f"Added a beacon at {self._beacon.position} with type \"{self._beacon.beacon_type}\""
        return f"Added a beacon at {self._beacon.position}"

    def undo(self, experiment: Experiment) -> str:
        experiment.beacons.remove(self._beacon.position)
        return f"Removed a beacon at {self._beacon.position}"


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


class _ChangeBeaconTypeAction(UndoableAction):
    _beacon: Position
    _old_type: Optional[str]
    _new_type: Optional[str]

    def __init__(self, beacon: Position, old_type: Optional[str], new_type: Optional[str]):
        self._beacon = beacon
        self._old_type = old_type
        self._new_type = new_type

    def do(self, experiment: Experiment) -> str:
        experiment.beacons.set_beacon_type(self._beacon, self._new_type)
        if self._new_type is None:
            return f"Removed the type of the beacon at {self._beacon}"
        return f"Set the type of the beacon {self._beacon} to \"{self._new_type}\""

    def undo(self, experiment: Experiment) -> str:
        experiment.beacons.set_beacon_type(self._beacon, self._old_type)
        if self._old_type is None:
            return f"Removed the type of the beacon at {self._beacon} again"
        return f"Changed the type of the beacon at {self._beacon} back to \"{self._old_type}\""


class BeaconEditor(AbstractEditor):
    """Editor for beacons - abstract points that cells move around or towards. Use Insert, Shift and Delete to insert,
    shift or delete points."""

    _selected_beacon_position: Optional[Position] = None
    _draw_beacon_distances: bool = False

    def __init__(self, window: Window):
        super().__init__(window, parent_viewer=LinkAndPositionEditor)

    def _selected_beacon(self) -> Optional[Beacon]:
        if self._selected_beacon_position is None:
            return None
        beacon_type = self._experiment.beacons.get_beacon_type(self._selected_beacon_position)
        return Beacon(position=self._selected_beacon_position, beacon_type=beacon_type)

    def _get_available_beacon_types(self) -> List[Marker]:
        return list(self._window.registry.get_registered_markers(Beacon))

    def get_extra_menu_options(self) -> Dict[str, Any]:
        menu_options = {
            **super().get_extra_menu_options(),
            "Edit//Positions-Toggle showing distances to positions": self._toggle_showing_beacon_distances
        }

        beacon_types = self._get_available_beacon_types()
        for beacon_type in beacon_types:
            menu_options[f"Edit//Beacons-Change type to \"{beacon_type.display_name}\""] = partial(self._set_beacon_type, beacon_type)
        if len(beacon_types) > 0:
            menu_options["Edit//Beacons-Remove type"] = lambda: self._set_beacon_type(None)

        return menu_options

    def _get_figure_title(self) -> str:
        return ("Editing beacons of time point " + str(self._time_point.time_point_number())
                + "    (z=" + self._get_figure_title_z_str() + ")")

    def _toggle_showing_beacon_distances(self):
        self._draw_beacon_distances = not self._draw_beacon_distances
        self.draw_view()

    def _set_beacon_type(self, new_type: Optional[Marker]):
        selected = self._selected_beacon()
        if selected is None:
            self.update_status("No beacon was selected - cannot change anything.")
            return
        old_type = selected.beacon_type
        new_type_str = new_type.save_name if new_type is not None else None
        self._perform_action(_ChangeBeaconTypeAction(selected.position, old_type, new_type_str))

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if not self._draw_beacon_distances or dt != 0 or abs(dz) > self.MAX_Z_DISTANCE:
            return super()._on_position_draw(position, color, dz, dt)

        resolution = self._experiment.images.resolution()
        beacon = self._experiment.beacons.find_closest_beacon(position, resolution)

        if beacon is None:
            return super()._on_position_draw(position, color, dz, dt)
        is_selected = beacon.beacon_position == self._selected_beacon_position

        background_color = (1, 1, 1, 0.8) if is_selected else (0, 1, 0, 0.8)
        self._draw_annotation(position, f"{beacon.distance_um:.1f} μm", background_color=background_color)
        return True

    def _on_mouse_single_click(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            super()._on_mouse_single_click(event)
            return

        clicked_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        resolution = self._experiment.images.resolution()
        closest_beacon = self._experiment.beacons.find_closest_beacon(clicked_position, resolution)
        if closest_beacon is None:
            self._selected_beacon_position = None
        elif closest_beacon.distance_um > 8:
            self._selected_beacon_position = None
            self.draw_view()
            self.update_status("Deselected the beacon.")
        else:
            self._selected_beacon_position = closest_beacon.beacon_position
            self.draw_view()
            self.update_status("Selected a beacon.")

    def _draw_extra(self):
        """Draws the selection box."""
        selected = self._selected_beacon()
        if selected is not None:
            beacon_name = selected.beacon_type
            font_weight = "bold"
            font_style = "normal"
            if beacon_name is None:
                # Unknown type
                beacon_name = "(untyped)"
                font_weight = "normal"
                font_style = "italic"
            else:
                # Try to replace with display name
                beacon_type_marker = self._window.registry.get_marker_by_save_name(beacon_name)
                if beacon_type_marker is not None and beacon_type_marker.applies_to(Beacon):
                    beacon_name = beacon_type_marker.display_name

            self._draw_selection(selected.position, core.COLOR_CELL_CURRENT)
            dz = int(abs(selected.position.z - self._z))
            if dz < 10:
                self._ax.annotate(beacon_name, (selected.position.x, selected.position.y), fontsize=8 - abs(dz / 2),
                                  fontweight=font_weight, fontstyle=font_style, color="black",
                                  backgroundcolor=(1, 1, 1, 0.8))

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

        current_selection = self._selected_beacon()
        beacon_type = None if current_selection is None else current_selection.beacon_type

        beacon_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        self._selected_beacon_position = beacon_position
        self._perform_action(_InsertBeaconAction(Beacon(position=beacon_position, beacon_type=beacon_type)))

    def _try_remove(self):
        selected = self._selected_beacon()
        if selected is None:
            self.update_status("No beacon was selected - cannot delete anything.")
            return
        self._selected_beacon_position = None
        self._perform_action(ReversedAction(_InsertBeaconAction(selected)))
