from typing import List, Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent

from autotrack import core
from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position
from autotrack.gui.window import Window
from autotrack.linking.nearby_position_finder import find_closest_position
from autotrack.visualizer import Visualizer, activate, DisplaySettings


class PositionListVisualizer(Visualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    _current_position_index: int = -1
    _position_list = List[Position]
    _show_next_image: bool

    __last_position_by_class = dict()  # Static variable

    def __init__(self, window: Window, all_positions: List[Position], chosen_position: Optional[Position] = None,
                 show_next_image: bool = False):
        """Creates a viewer for a list of positions. The positions will automatically be sorted by time_point number.
        chosen_position is a position that is used as a starting point for the viewer, but only if it appears in the
        list
        """
        super().__init__(window)
        self._position_list = all_positions
        self._position_list.sort(key=lambda position: position.time_point_number())
        self._show_closest_or_stored_position(chosen_position)  # Calling a self.method during construction is bad...
        self._show_next_image = show_next_image

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View//Exit-Exit this view (Esc)": lambda: self._exit_view(),
            "Navigate//Time-Next (Right)": self._goto_next,
            "Navigate//Time-Previous (Left)": self._goto_previous
        }

    def _show_closest_or_stored_position(self, position: Optional[Position]):
        if position is not None:
            # Try to find selected position
            try:
                self._current_position_index = self._position_list.index(position)
            except ValueError:
                # Try nearest position
                close_match = find_closest_position(self._position_list, position, max_distance=100)

                if close_match is not None and close_match.time_point_number() == position.time_point_number():
                    self._current_position_index = self._position_list.index(close_match)
                    return

        # Give up, show position from before
        try:
            position = self._get_last_position()
            self._current_position_index = self._position_list.index(position)
        except ValueError:
            pass  # Ignore, last position is no longer avalable

    def _get_last_position(self):
        """Gets the index we were at last time a visualizer of this kind was open."""
        try:
            return PositionListVisualizer.__last_position_by_class[type(self)]
        except KeyError:
            return -1

    def get_message_no_positions(self):
        return "No cells found. Is there some data missing?"

    def get_message_press_right(self):
        return "Press right to view the first cell."

    def draw_view(self):
        self._clear_axis()
        if self._current_position_index < 0 or self._current_position_index >= len(self._position_list):
            if len(self._position_list) == 0:
                self._window.set_figure_title(self.get_message_no_positions())
            else:
                self._window.set_figure_title(self.get_message_press_right())
            self._fig.canvas.draw()
            return

        self._zoom_to_cell()
        self._show_image()

        current_position = self._position_list[self._current_position_index]
        shape = self._experiment.positions.get_shape(current_position)
        shape.draw2d(current_position.x, current_position.y, 0, 0, self._ax, core.COLOR_CELL_CURRENT)
        self._draw_connections(self._experiment.links, current_position)
        self._window.set_figure_title(self.get_title(self._position_list, self._current_position_index))

        self._fig.canvas.draw()
        PositionListVisualizer.__last_position_by_class[type(self)] = current_position

    def _zoom_to_cell(self):
        mother = self._position_list[self._current_position_index]
        self._ax.set_xlim(mother.x - 50, mother.x + 50)
        self._ax.set_ylim(mother.y + 50, mother.y - 50)
        self._ax.set_autoscale_on(False)

    def _draw_connections(self, links: PositionLinks, main_position: Position, line_style:str = "solid",
                          line_width: int = 1):
        for connected_position in links.find_links_of(main_position):
            delta_time = 1
            if connected_position.time_point_number() < main_position.time_point_number():
                delta_time = -1
                if self._show_next_image:
                    continue  # Showing the previous position only makes things more confusing here

            color = core.COLOR_CELL_NEXT if delta_time == 1 else core.COLOR_CELL_PREVIOUS
            position_shape = self._experiment.positions.get_shape(connected_position)

            self._ax.plot([connected_position.x, main_position.x], [connected_position.y, main_position.y],
                          color=color, linestyle=line_style, linewidth=line_width)
            position_shape.draw2d(connected_position.x, connected_position.y, 0, delta_time, self._ax, color)

    def _show_image(self):
        mother = self._position_list[self._current_position_index]
        time_point = self._experiment.get_time_point(mother.time_point_number())
        image_stack = self.load_image(time_point, self._show_next_image)
        if image_stack is not None:
            z = max(0, min(int(mother.z), len(image_stack) - 1))
            self._ax.imshow(image_stack[z], cmap="gray")

    def _goto_next(self):
        self._current_position_index += 1
        if self._current_position_index >= len(self._position_list):
            self._current_position_index = 0
        self.draw_view()

    def _goto_previous(self):
        self._current_position_index -= 1
        if self._current_position_index < 0:
            self._current_position_index = len(self._position_list) - 1
        self.draw_view()

    def _exit_view(self):
        from autotrack.visualizer.standard_image_visualizer import StandardImageVisualizer

        if self._current_position_index < 0 or self._current_position_index >= len(self._position_list):
            # Don't know where to go
            image_visualizer = StandardImageVisualizer(self._window, display_settings=
                                                       DisplaySettings(show_next_time_point=self._show_next_image))
        else:
            mother = self._position_list[self._current_position_index]
            image_visualizer = StandardImageVisualizer(self._window,
                                                       time_point_number=mother.time_point_number(), z=int(mother.z),
                                                       display_settings=DisplaySettings(
                                                           show_next_time_point=self._show_next_image))
        activate(image_visualizer)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "left":
            self._goto_previous()
        elif event.key == "right":
            self._goto_next()
        elif event.key == DisplaySettings.KEY_SHOW_NEXT_IMAGE_ON_TOP:
            self._show_next_image = not self._show_next_image
            self.draw_view()
        elif event.key == "escape":
            self._exit_view()

    def _on_command(self, command: str) -> bool:
        if command == "exit":
            self._exit_view()
            return True
        if command == "help":
            self.update_status("Available commands:\n"
                               "/exit - Exits this view, and goes back to the main view.")
            return True
        return super()._on_command(command)

    def get_title(self, all_cells: List[Position], cell_index: int):
        mother = all_cells[cell_index]
        return "Cell " + str(self._current_position_index + 1) + "/" + str(len(self._position_list)) + "\n" + str(mother)
