import numpy
import skimage

from organoid_tracker.core import TimePoint
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer.image_offset_editor import _ChangeAllPositionsAction


def get_menu_items(window: Window):
    return {
        "Edit//Batch-Automatic offsets...":
            lambda: _automatically_offset_images(window),
    }

def _automatically_offset_images(window: Window):

    scale_down = [0.5, 0.2, 0.2]

    for tab in window.get_gui_experiment().get_active_tabs():
        from skimage.registration import phase_cross_correlation
        from skimage.transform import rescale

        min_t = tab.experiment.first_time_point_number()
        max_t = tab.experiment.last_time_point_number()

        offset_list = [Position(0,0,0, time_point_number=min_t)]
        offset = numpy.array([0,0,0])

        for t in range(min_t, max_t):

            image_1 = tab.experiment.images.get_image_stack(TimePoint(t))
            image_2 = tab.experiment.images.get_image_stack(TimePoint(t + 1))

            image_1 = rescale(image_1, scale_down)
            image_2 = rescale(image_2, scale_down)

            mask = numpy.zeros_like(image_1)
            mask[5:-5, 5:-5, 5:-5] = 1

            shift, error, phasediff = phase_cross_correlation(image_1, image_2, reference_mask= mask)

            print(shift)
            print(error)

            #shift, error, phasediff = phase_cross_correlation(image_1[40,...], image_2[40,...], reference_mask=mask[40,...])

            #shift = numpy.array([0,shift[0],shift[1]])

            print(t)
            print(shift)

            offset = offset + shift/scale_down

            offset_list.append(Position(offset[2], offset[1], offset[0], time_point_number=t+1))

        offsets = ImageOffsets(offset_list)

        result_message = tab.undo_redo.do(_ChangeAllPositionsAction(tab.experiment.images.offsets, offsets), tab.experiment)
        window.set_status(result_message)
