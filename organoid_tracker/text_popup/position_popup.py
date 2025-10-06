from typing import Optional

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup


class PositionPopup(RichTextPopup):
    _window: Window
    _position: Position

    def __init__(self, window: Window, position: Position):
        self._window = window
        self._position = position

    def get_title(self) -> str:
        return str(self._position)

    def navigate(self, url: str) -> Optional[str]:
        if url == self.INDEX:
            return self._get_position_page()

        if url.startswith("goto "):
            parts = url.split(" ")
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            t = int(parts[4])
            self._window.get_gui_experiment().goto_position(Position(x, y, z, time_point_number=t))
            return None
        raise UserError("Page not found", "Page not found: " + url)

    def _get_position_page(self) -> str:
        try:
            experiment = self._window.get_experiment()
        except UserError:
            experiment = Experiment()
        text = f"# {self._position}\n\n"
        text += f"[Go to position](goto {self._position.x} {self._position.y} {self._position.z} {self._position.time_point_number()})"

        text += "\n\n## Metadata\n\nAny metadata stored on the position\n\n"
        for name, value in experiment.positions.find_all_data_of_position(self._position):
            text += f"* **{name}:** `{value}`\n"

        text += "\n\n## Links\n\nShows the links over time.\n\n"
        for i, position in enumerate(experiment.links.find_links_of(self._position)):
            text += f"* Towards {position} [View](goto {position.x} {position.y} {position.z} {position.time_point_number()})  \n"

            link_data = dict(experiment.links.find_all_data_of_link(self._position, position))
            if len(link_data) > 0:
                text += f"  \n  Link metadata: `{link_data}`\n\n"
        if len(experiment.links.find_links_of(self._position)) == 0:
            text += "*Position has no links.*"

        text += "\n\n## Connections\n\nRelated positions in the same time point.\n\n"
        for i, position in enumerate(experiment.connections.find_connections(self._position)):
            text += f"* Towards {position} [View](goto {position.x} {position.y} {position.z} {position.time_point_number()})\n"
        if len(list(experiment.connections.find_connections(self._position))) == 0:
            text += "*Position has no connections.*"

        return text