from typing import Dict, Any

import numpy

from autotrack.gui.window import Window
from autotrack.imaging import angles
from autotrack.position_analysis import cell_curvature_calculator
from autotrack.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Bla//Curvature-Curvature test": lambda: _show_pairs(window)
    }


def _show_pairs(window: Window):
    axes = window.get_figure().gca()
    print("--")
    position = None
    i = 0
    for a_position in window.get_experiment().positions:
        position = a_position
        if i > 3:
            break
        i += 1
    window.get_gui_experiment().goto_position(position)

    i = 0
    found_angles = list()
    resolution = window.get_experiment().images.resolution()
    for position1, position2 in cell_curvature_calculator.get_curvature_pairs(window.get_experiment(), position):
        axes.plot([position1.x, position.x, position2.x], [position1.y, position.y, position2.y], color=SANDER_APPROVED_COLORS[i], linewidth=5)
        found_angle = angles.right_hand_rule(position1.to_vector_um(resolution), position.to_vector_um(resolution), position2.to_vector_um(resolution))
        found_angles.append(found_angle)
        #print(position1, position2)
        i += 1
    found_angles = numpy.array(found_angles)
    print(found_angles, "avg", found_angles.mean())
    window.get_figure().canvas.draw()
