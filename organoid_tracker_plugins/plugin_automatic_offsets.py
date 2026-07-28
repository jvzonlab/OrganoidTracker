from typing import Iterable, Any

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, worker_job
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.gui.worker_job import WorkerJob
from organoid_tracker.position_detection import position_mover

_SCALE_DOWN = [0.5, 0.2, 0.2]


class _ChangeOffsetsAction(UndoableAction):
    _offsets_before: ImageOffsets
    _offsets_after: ImageOffsets

    def __init__(self, offsets_before: ImageOffsets, offsets_after: ImageOffsets):
        self._offsets_before = offsets_before
        self._offsets_after = offsets_after

    def do(self, experiment: Experiment) -> str:
        experiment.images.offsets = self._offsets_after
        position_mover.update_positions_for_changed_offsets(experiment, self._offsets_before)
        return "Moved all positions, links and images using the phase cross-correlation estimation.."

    def undo(self, experiment: Experiment) -> str:
        experiment.images.offsets = self._offsets_before
        position_mover.update_positions_for_changed_offsets(experiment, self._offsets_after)
        return "Moved all positions, links and images  to what they were."


def get_menu_items(window: Window):
    return {
        "Edit//Batch-Automatic offsets...":
            lambda: _automatically_offset_images(window),
    }


def _automatically_offset_images(window: Window):
    if not dialog.popup_message_cancellable("Automatic offsets", "This will calculate image offsets for"
        " all experiments based on the phase cross-correlation of the images in the current channel."
        " This may take a while. Continue?"):
        return

    image_channel = window.display_settings.image_channel
    worker_job.submit_job(window, _AutomaticOffsetCalculator(image_channel))


class _AutomaticOffsetCalculator(WorkerJob):

    _channel: ImageChannel

    def __init__(self, channel: ImageChannel):
        self._channel = channel

    def copy_experiment(self, experiment: Experiment) -> Experiment:
        return experiment.copy_selected(images=True)

    def gather_data(self, experiment_copy: Experiment) -> Any:
        from skimage.registration import phase_cross_correlation
        from skimage.transform import rescale

        min_t = experiment_copy.first_time_point_number()
        max_t = experiment_copy.last_time_point_number()
        if min_t is None or max_t is None:
            return ImageOffsets()  # Experiment doesn't have images, skip

        offset_list = [Position(0, 0, 0, time_point_number=min_t)]
        offset = numpy.array([0, 0, 0])

        time_points = [TimePoint(t) for t in range(min_t, max_t)]
        for time_point in self.reporting_progress(time_points):
            image_1 = experiment_copy.images.get_image_stack(time_point, self._channel)
            image_2 = experiment_copy.images.get_image_stack(time_point + 1, self._channel)
            if image_1 is None or image_2 is None:
                continue  # Time point is missing

            image_1 = rescale(image_1, _SCALE_DOWN)
            image_2 = rescale(image_2, _SCALE_DOWN)

            mask = numpy.zeros_like(image_1)
            mask[5:-5, 5:-5, 5:-5] = 1

            shift, error, phasediff = phase_cross_correlation(image_1, image_2, reference_mask=mask)

            # print(shift)
            # print(error)

            # shift, error, phasediff = phase_cross_correlation(image_1[40,...], image_2[40,...], reference_mask=mask[40,...])
            # shift = numpy.array([0,shift[0],shift[1]])
            # print(t)
            # print(shift)

            offset = offset + shift / _SCALE_DOWN

            offset_list.append(Position(offset[2], offset[1], offset[0], time_point_number=time_point.time_point_number() + 1))

        return ImageOffsets(offset_list)

    def use_data(self, tab: SingleGuiTab, data: Any):
        offsets: ImageOffsets = data
        tab.undo_redo.do(_ChangeOffsetsAction(tab.experiment.images.offsets, offsets), tab.experiment)

    def on_finished(self, data: Iterable[Any]):
        dialog.popup_message("Calculations done", "Finished calculating offsets for all experiments.")

