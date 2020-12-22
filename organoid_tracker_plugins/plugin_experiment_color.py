from typing import Dict, Any, Optional

from organoid_tracker.core import Color, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Experiment-Set color...":
            lambda: _show_color_prompt(window)
    }


class _ColorPopup(RichTextPopup):

    _window: Window

    def __init__(self, window: Window):
        self._window = window

    def get_title(self) -> str:
        return "Color chooser"

    def navigate(self, url: str) -> Optional[str]:
        try:
            experiment = self._window.get_experiment()
        except UserError as e:
            return "# " + e.title + "\n\n" + e.body

        if url == self.INDEX:
            return f"""
# Color of the experiment

<p style=\"background:{experiment.color.to_html_hex()};width=40px;height=40px\">
    &nbsp;<br>&nbsp;<br>&nbsp;
</p>

<a href="change">Change</a>

The color is used in certain graphs that show multiple experiments, so that each experiment can be recognized. In
addition, it ensures that each experiment has a consistent color across different graphs. If you're writing your own
graph, you can access the color from Python as `experiment.color`.
            """
        if url == "change":
            new_color = dialog.prompt_color("Color", default_color=experiment.color)
            if new_color is not None:
                self._window.perform_data_action(_ColorChangeAction(new_color))
                return self.navigate(self.INDEX)
        return None


class _ColorChangeAction(UndoableAction):
    """Action to change the color of an experiment."""

    _old_color: Optional[Color] = None
    _new_color: Color

    def __init__(self, color: Color):
        self._new_color = color

    def do(self, experiment: Experiment) -> str:
        self._old_color = experiment.color
        experiment.color = self._new_color
        return f"Changed the color of the experiment to {self._new_color}"

    def undo(self, experiment: Experiment) -> str:
        experiment.color = self._old_color
        return f"Changed the color of the experiment back to {self._old_color}"


def _show_color_prompt(window: Window):
    window.get_experiment()  # Ensures an experiment is selected
    dialog.popup_rich_text(_ColorPopup(window))
