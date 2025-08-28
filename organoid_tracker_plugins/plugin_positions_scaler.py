from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window):
    return {
        "Edit//Tracking-Scale positions...": lambda: _scale_positions(window),
    }


def _scale_positions(window: Window):
    """in x, y and z"""
    xy_scale = dialog.prompt_float("XY scaling", "By what factors should the positions be scaled?", minimum=0.1,
                                   default=1)

    if xy_scale is None:
        return

    z_scale = dialog.prompt_float("Z scaling", "By what factors should the positions be scaled?", minimum=0.1,
                                  default=1)
    if z_scale is None:
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment

        # Scale all positions
        for time_point in experiment.positions.time_points():
            for position in list(experiment.positions.of_time_point(time_point)):
                new_position = Position(position.x * xy_scale, position.y * xy_scale, position.z * z_scale,
                                        time_point=position.time_point())
                experiment.move_position(position, new_position, update_splines=False)
            experiment.splines.update_for_changed_positions(time_point,
                                                            list(experiment.positions.of_time_point(time_point)))

        # Scale offsets
        new_offsets = list()
        for time_point in experiment.time_points():
            offset = experiment.images.offsets.of_time_point(time_point)
            new_offset = Position(offset.x * xy_scale, offset.y * xy_scale, offset.z * z_scale,
                                  time_point=time_point)
            new_offsets.append(new_offset)
        experiment.images.offsets = ImageOffsets(new_offsets)

    dialog.popup_message("Scaling", "Scaled all positions!")
