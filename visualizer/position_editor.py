from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent
import core
from core import Particle
from gui import Window
from visualizer import DisplaySettings, activate
from visualizer.image_visualizer import AbstractImageVisualizer


class PositionEditor(AbstractImageVisualizer):
    """Editor for positions.
    Adding a cell: press Insert to add a cell at your mouse position
    Moving a cell: double-click on a cell, then press Shift to move to cell to your mouse position
    Removing a cell: double-click on a cell, then press Delete"""

    _selected: Optional[Particle] = None

    def __init__(self, window: Window, time_point_number: int = 1, z: int = 14):
        super().__init__(window, time_point_number, z, DisplaySettings(show_shapes=False))

    def get_extra_menu_options(self):
        parent = super().get_extra_menu_options()
        view = parent.get("View", [])

        return {
            **parent,
            "View": [
                *view,
                '-',
                ("Exit this view", self._exit_view)
            ],
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert":
            self._time_point.add_particle(Particle(event.xdata, event.ydata, self._z))
            self.draw_view()
            self._update_status("Added cell at x,y,z = " + str(event.xdata) + "," + str(event.ydata) + "," + str(self._z))
        elif event.key == "delete":
            self._try_delete()
        elif event.key == "shift":
            self._try_move(event.xdata, event.ydata)
        elif event.key == "p":
            self._exit_view()
        else:
            super()._on_key_press(event)

    def _try_delete(self):
        if self._selected is None:
            self._update_status("Cannot delete anything: no cell selected")
            return
        if self._selected.time_point_number() != self._time_point.time_point_number():
            self._update_status("Cannot delete cell from another time point")
            return
        self._experiment.remove_particle(self._selected)
        self._update_status("Deleted " + str(self._selected))
        self._selected = None
        self.draw_view()

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return

        new_selection = self._get_particle_at(event.xdata, event.ydata)
        if new_selection is None:
            self._update_status("Cannot find a particle here")
            return

        if new_selection == self._selected:
            self._selected = None  # Deselect
            self._update_status("Deselected " + str(new_selection))
        else:
            self._selected = new_selection  # Select
            self._update_status("Selected " + str(new_selection))
        self.draw_view()

    def _draw_extra(self):
        self._draw_highlight(self._selected)

    def _draw_highlight(self, particle: Optional[Particle]):
        if particle is None:
            return
        color = core.COLOR_CELL_CURRENT
        if particle.time_point_number() < self._time_point.time_point_number():
            color = core.COLOR_CELL_PREVIOUS
        elif particle.time_point_number() > self._time_point.time_point_number():
            color = core.COLOR_CELL_NEXT
        self._ax.plot(particle.x, particle.y, 'o', markersize=25, color=(0,0,0,0), markeredgecolor=color,
                      markeredgewidth=5)

    def _try_move(self, x: float, y: float):
        if self._selected is None:
            self._update_status("Cannot move anything: no cell selected")
            return
        if self._selected.time_point_number() != self._time_point.time_point_number():
            self._update_status("Cannot move cell from another time point")
            return
        if abs(self._selected.x - x) < 0.01 and abs(self._selected.y - y) < 0.01 \
                and abs(self._selected.z - self._z) < 0.01:
            self._update_status("Cell didn't move. "
                                "Hold your mouse at the position where you want the cell to go to.")
            return
        new_position = Particle(x, y, self._z)
        if self._experiment.move_particle(self._selected, new_position):
            self._selected = new_position
            self.draw_view()
            self._update_status("Cell shifted to " + str((x, y, self._z)))

    def _exit_view(self):
        from visualizer.image_visualizer import StandardImageVisualizer
        image_visualizer = StandardImageVisualizer(self._window, time_point_number=self._time_point.time_point_number(),
                                                   z=self._z, display_settings=self._display_settings)
        activate(image_visualizer)
