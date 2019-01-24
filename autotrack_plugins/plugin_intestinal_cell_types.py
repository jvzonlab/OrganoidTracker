"""Registers the Stem and Paneth cell type."""

from autotrack.core.position import PositionType
from autotrack.gui.window import Window


STEM = PositionType("STEM", "stem cell", (255, 0, 0))
PANETH = PositionType("PANETH", "Paneth cell", (0, 0, 255))


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    gui_experiment.register_position_type(PANETH)
