"""Registers the Stem and Paneth cell type."""
from autotrack.core.data_axis import DataAxis
from autotrack.core.position import Position
from autotrack.core.marker import Marker
from autotrack.gui.window import Window


STEM = Marker([Position], "STEM", "stem cell", (255, 0, 0))  # Not yet used or registered
PANETH = Marker([Position], "PANETH", "Paneth cell", (0, 0, 255))

CRYPT = Marker([DataAxis], "CRYPT", "crypt axis", (255, 0, 0))
VILLUS = Marker([DataAxis], "VILLUS", "villus region", (255, 0, 0))


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    gui_experiment.register_marker(PANETH)

    gui_experiment.register_marker(CRYPT)
    gui_experiment.register_marker(VILLUS)
