"""Registers the Stem and Paneth cell type."""
from autotrack.core.spline import Spline
from autotrack.core.position import Position
from autotrack.core.marker import Marker
from autotrack.gui.window import Window


STEM = Marker([Position], "STEM", "stem cell", (255, 227, 190))

# Differentiated
M_CELL = Marker([Position], "M_CELL", "M cell", (242, 109, 84))
ENTEROCYTE = Marker([Position], "ENTEROCYTE", "enterocyte cell", (251, 175, 64))
PANETH = Marker([Position], "PANETH", "Paneth cell", (63, 172, 225))
GOBLET = Marker([Position], "GOBLET", "Goblet cell", (147, 190, 147))
ENTEROENDOCRINE = Marker([Position], "ENTEROENDOCRINE", "enteroendocrine cell", (239, 212, 202))
TUFT = Marker([Position], "TUFT", "Tuft cell", (101, 46, 143))

CRYPT = Marker([Spline], "CRYPT", "crypt axis", (255, 0, 0), is_axis=True)


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    for marker in [STEM, M_CELL, ENTEROCYTE, PANETH, GOBLET, ENTEROENDOCRINE, TUFT]:
        gui_experiment.register_marker(marker)

    gui_experiment.register_marker(CRYPT)
