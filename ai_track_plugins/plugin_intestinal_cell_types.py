"""Registers the Stem and Paneth cell type."""
from ai_track.core.spline import Spline
from ai_track.core.position import Position
from ai_track.core.marker import Marker
from ai_track.gui.window import Window


STEM = Marker([Position], "STEM", "stem cell", (139, 79, 68))

# Differentiated
M_CELL = Marker([Position], "M_CELL", "M cell", (242, 109, 84))
ENTEROCYTE = Marker([Position], "ENTEROCYTE", "enterocyte cell", (207, 228, 243))
LUMEN = Marker([Position], "LUMEN", "lumen", (200, 200, 200))
PANETH = Marker([Position], "PANETH", "Paneth cell", (115, 220, 113))
GOBLET = Marker([Position], "GOBLET", "Goblet cell", (210, 162, 47))
ENTEROENDOCRINE = Marker([Position], "ENTEROENDOCRINE", "enteroendocrine cell", (108, 92, 231))
TUFT = Marker([Position], "TUFT", "Tuft cell", (101, 46, 143))

CRYPT = Marker([Spline], "CRYPT", "crypt axis", (255, 0, 0), is_axis=True)


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    for marker in [STEM, M_CELL, ENTEROCYTE, LUMEN, PANETH, GOBLET, ENTEROENDOCRINE, TUFT]:
        gui_experiment.register_marker(marker)

    gui_experiment.register_marker(CRYPT)
