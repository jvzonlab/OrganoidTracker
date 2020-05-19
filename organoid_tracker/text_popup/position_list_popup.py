from typing import List, Optional

from organoid_tracker.core import UserError
from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup


class PositionListPopup(RichTextPopup):
    """Used to display a list of positions that the user can visit by clicking on it. Use dialog.popup_rich_text(...)
     to show this popup"""

    _window: Window
    _title: str
    _description: str
    _position_descriptions: List[str]
    _positions: List[Position]

    def __init__(self, window: Window, *, title: str, positions: List[Position],
                 position_descriptions: Optional[List[str]] = None, description: str = ""):
        self._window = window
        self._title = title
        self._description = description
        self._positions = positions
        if position_descriptions is None:
            self._position_descriptions = [""] * len(positions)
        else:
            self._position_descriptions = position_descriptions

    def get_title(self) -> str:
        return self._title

    def navigate(self, url: str) -> Optional[str]:
        if url == self.INDEX:
            text = "# " + self._title + "\n\n" + self._description + "\n\n"
            for i, position in enumerate(self._positions):
                description = self._position_descriptions[i]
                text += f"{i + 1}. [View](goto {position.x} {position.y} {position.z} {position.time_point_number()}) {position} {description}\n"
            return text
        if url.startswith("goto "):
            parts = url.split(" ")
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            t = int(parts[4])
            self._window.get_gui_experiment().goto_position(Position(x, y, z, time_point_number=t))
            return None
        raise UserError("Page not found", "Page not found: " + url)
