import math
from functools import partial
from typing import Optional, Dict, Any, Tuple, List

import PIL.Image
import PIL.ImageDraw
import numpy
from matplotlib.backend_bases import MouseEvent, KeyEvent
from numpy import ndarray

from organoid_tracker import core
from organoid_tracker.core import UserError, image_coloring, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction, UndoRedo
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_editor import AbstractEditor
from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor


class _DeleteLabelAction(UndoableAction):
    """Action that deletes a label from a segmentation stack."""
    _segmentation_stack: ndarray
    _label: int

    _old_coords: Optional[ndarray] = None  # Coordinates of the deleted label, filled in do()

    def __init__(self, segmentation_stack: ndarray, label: int):
        self._segmentation_stack = segmentation_stack
        self._label = label

    def do(self, experiment: Experiment) -> str:
        self._old_coords = self._segmentation_stack == self._label

        self._segmentation_stack[self._old_coords] = 0
        return "Deleted label " + str(self._label)

    def undo(self, experiment: Experiment) -> str:
        self._segmentation_stack[self._old_coords] = self._label
        return "Restored label " + str(self._label)


class _SetLabelAction(UndoableAction):
    """Action that adds a label to a segmentation stack."""
    _segmentation_stack: ndarray
    _mask: ndarray
    _mask_x_start: int
    _mask_y_start: int
    _mask_z: int
    _label_to_modify: int
    _delete: bool

    _old_crop: Optional[ndarray] = None  # Cropped area of the segmentation stack, filled in do()

    def __init__(self, segmentation_stack: ndarray, mask: ndarray, mask_x_start: int, mask_y_start: int, mask_z: int,
                 label_to_modify: int, delete: bool):
        """Creates a new action that adds or removes a label from a segmentation stack. Note: the mask must be fully
        within the segmentation stack."""
        self._segmentation_stack = segmentation_stack
        self._mask = mask
        self._mask_x_start = mask_x_start
        self._mask_y_start = mask_y_start
        self._mask_z = mask_z
        self._label_to_modify = label_to_modify
        self._delete = delete

    def do(self, experiment: Experiment) -> str:
        width = self._mask.shape[1]
        height = self._mask.shape[0]

        segmentation_crop = self._segmentation_stack[self._mask_z, self._mask_y_start:self._mask_y_start + height,
                            self._mask_x_start:self._mask_x_start + width]
        self._old_crop = segmentation_crop.copy()
        if self._delete:
            # Remove parts of the mask that don't match the label
            self._mask[segmentation_crop != self._label_to_modify] = 0

            # Then remove the mask from the segmentation
            segmentation_crop[self._mask > 0] = 0
            return "Deleted (part of) label " + str(self._label_to_modify)
        else:
            segmentation_crop[self._mask > 0] = self._label_to_modify
            return "Added/expanded segmentation mask with label " + str(self._label_to_modify)

    def undo(self, experiment: Experiment) -> str:
        self._segmentation_stack[self._mask_z, self._mask_y_start:self._mask_y_start + self._mask.shape[0],
        self._mask_x_start:self._mask_x_start + self._mask.shape[1]] = self._old_crop
        self._old_crop = None
        return "Restored segmentation mask"


def _find_new_label(segmentation_image: ndarray) -> int:
    """Returns a new label that is not yet used in the segmentation image."""
    used_labels = numpy.unique(segmentation_image)
    new_label = 1
    while new_label in used_labels:
        new_label += 1
    return new_label


class CellSegmentationEditor(AbstractEditor):
    """Editor for cell segmentation images. Note that this editor overwrites images in the folder. To get started,
    select a channel to edit in the Edit menu."""
    _selected_position: Optional[Position] = None

    # Information on the currently loaded segmentation image
    _segmentation_image_experiment: Optional[Experiment] = None
    _segmentation_image_channel: Optional[ImageChannel] = None
    _segmentation_image_stack: Optional[ndarray] = None
    _segmentation_image_time_point: Optional[TimePoint] = None

    # We use a separate undo/redo queue for the segmentation images, as actions are only valid for the current time
    # point. So we need to clear the queue when switching time points.
    _segmentation_image_undo_redo: UndoRedo

    _clicked_positions: List[Position]

    def __init__(self, window: Window, selected_position: Optional[Position] = None):
        super().__init__(window, LinkAndPositionEditor)
        self._clicked_positions = []
        self._segmentation_image_undo_redo = UndoRedo()
        self._selected_position = selected_position

    def _calculate_time_point_metadata(self):
        # Overridden to update the segmentation image
        super()._calculate_time_point_metadata()

        # Try to move selection to current time point
        if self._selected_position is not None:
            moved_position = self._experiment.links.get_position_at_time_point(self._selected_position, self._time_point)
            if moved_position is not None:
                self._selected_position = moved_position

        self._update_segmentation_image()

    def _return_2d_image(self, time_point: TimePoint, z: int, channel: ImageChannel, show_next_time_point: bool) -> Optional[ndarray]:
        # Overridden to return the segmentation image if available
        # (the one in the Experiment object might be outdated)

        if time_point == self._segmentation_image_time_point and channel == self._display_settings.segmentation_channel:
            # Note: the show_next_time_point parameter is not implemented here
            # But the two-time point viewer is not useful for segmentation images anyways, as it forces a
            # magenta/green colormap
            image_z = z - int(self._experiment.images.offsets.of_time_point(time_point).z)
            if 0 <= image_z < self._segmentation_image_stack.shape[0]:
                return self._segmentation_image_stack[z]

        return super()._return_2d_image(time_point, z, channel, show_next_time_point)

    def _return_3d_image(self, time_point: TimePoint, channel: ImageChannel, show_next_time_point: bool) -> Optional[ndarray]:
        # Overridden to return the segmentation image if available
        # (the one in the Experiment object might be outdated)
        if time_point == self._segmentation_image_time_point and channel == self._display_settings.segmentation_channel:
            # Note: the show_next_time_point parameter is not implemented here
            # But the two-time point viewer is not useful for segmentation images anyways, as it forces a
            # magenta/green colormap
            return self._segmentation_image_stack

        return super()._return_3d_image(time_point, channel, show_next_time_point)

    def _update_segmentation_image(self):
        if self._segmentation_image_stack is not None:
            # Check if we have an outdated segmentation image in memory, if yes, save it to disk
            if self._time_point != self._segmentation_image_time_point or \
                    self._display_settings.segmentation_channel != self._segmentation_image_channel or \
                    self._experiment != self._segmentation_image_experiment:
                if self._has_unsaved_segmentation_changes():
                    self._save_segmentation_image()

        # Then, load a new segmentation image stack if needed
        if self._display_settings.segmentation_channel is None:
            # No segmentation channel available (anymore) - need to clear the segmentation image
            self._segmentation_image_stack = None
            self._segmentation_image_time_point = None
            self._segmentation_image_experiment = None
            self._segmentation_image_channel = None
            self._segmentation_image_undo_redo.clear()
            self._segmentation_image_undo_redo.mark_everything_saved()
        else:
            # Need to make sure that the right segmentation image is loaded
            if self._time_point != self._segmentation_image_time_point or \
                    self._display_settings.segmentation_channel != self._segmentation_image_channel or \
                    self._experiment != self._segmentation_image_experiment:
                # If we're here, the wrong one (or none at all) was loaded
                self._segmentation_image_stack = self._experiment.images.get_image_stack(self._time_point,
                                                                                         self._display_settings.segmentation_channel)
                if self._segmentation_image_stack is not None:
                    # Loaded an image, update bookkeeping
                    self._segmentation_image_time_point = self._time_point
                    self._segmentation_image_channel = self._display_settings.segmentation_channel
                    self._segmentation_image_experiment = self._experiment
                else:
                    # Failed to load image
                    self._segmentation_image_time_point = None
                    self._segmentation_image_channel = None
                    self._segmentation_image_experiment = None

                # Then, after loading the image, clear the undo/redo queue, as it was only valid for the previously
                # loaded image
                self._segmentation_image_undo_redo.clear()
                self._segmentation_image_undo_redo.mark_everything_saved()

    def _save_segmentation_image(self):
        if self._segmentation_image_stack is None or self._segmentation_image_experiment is None or \
                self._segmentation_image_channel is None or self._segmentation_image_time_point is None:
            return  # Nothing to save
        try:
            # Time to save the segmentation image stack
            self._segmentation_image_experiment.images.save_3d_image_array(
                self._segmentation_image_time_point,
                self._segmentation_image_channel,
                self._segmentation_image_stack)
        finally:
            self._segmentation_image_undo_redo.mark_everything_saved()

    def _get_segmentation_stack_and_selected_label(self) -> Tuple[Optional[ndarray], Optional[int]]:
        """Returns the segmentation stack and the label of the selected position, or None, None if no segmentation stack
        is available or no position is selected."""
        if self._selected_position is None or self._selected_position.time_point() != self._time_point:
            return None, None
        segmentation_stack = self._segmentation_image_stack
        if segmentation_stack is None:
            return None, None

        offset = self._experiment.images.offsets.of_time_point(self._time_point)
        image_position = self._selected_position - offset
        x, y, z = int(round(image_position.x)), int(round(image_position.y)), int(round(image_position.z))
        if 0 <= x < segmentation_stack.shape[2] and 0 <= y < segmentation_stack.shape[1] and 0 <= z < \
                segmentation_stack.shape[0]:
            return segmentation_stack, segmentation_stack[z, y, x]

        return None, None

    def _undo(self):
        # Redirect the undo function to our own undo/redo queue
        status = self._segmentation_image_undo_redo.undo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _redo(self):
        # Redirect the redo function to our own undo/redo queue
        status = self._segmentation_image_undo_redo.redo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _get_figure_title(self) -> str:
        return ("Editing segmentation masks of time point " + str(self._time_point.time_point_number())
                + "    (z=" + self._get_figure_title_z_str() + ")")

    def _exit_view(self):
        if self._selected_position is not None and self._selected_position.time_point() == self._time_point:
            # Just deselect, don't fully exit yet
            self._selected_position = None
            self.draw_view()
            self.update_status("Deselected current position.")
        else:
            # Save and exit
            self._force_exit_to_tracking_editor()

    def _force_exit_to_tracking_editor(self):
        # Save and exit, keeping the selection
        if self._has_unsaved_segmentation_changes():
            self._save_segmentation_image()

        selected_positions = [self._selected_position] if self._selected_position is not None else []
        image_visualizer = LinkAndPositionEditor(self._window, selected_positions=selected_positions)
        activate(image_visualizer)

    def _on_program_close(self):
        # Used to save the segmentation image when the program is closed
        if self._has_unsaved_segmentation_changes():
            self._save_segmentation_image()

    def _draw_extra(self):
        # Highlight segmentation mask
        segmentation_stack, label = self._get_segmentation_stack_and_selected_label()
        if segmentation_stack is not None and label is not None:
            self._draw_selected_mask(segmentation_stack, label)

        # Highlight selected position
        if self._selected_position is not None and self._selected_position.time_point() == self._time_point:
            self._draw_selection(self._selected_position, core.COLOR_CELL_CURRENT)

        # Draw drawing shape
        self._draw_drawing_shape()

    def _draw_selected_mask(self, segmentation_stack: ndarray, label: int):
        if label == 0:
            return

        offset = self._experiment.images.offsets.of_time_point(self._time_point)
        viewing_z = int(self._z - offset.z)
        if viewing_z < 0 or viewing_z >= segmentation_stack.shape[0]:
            return
        segmentation_slice_at_viewing_z = segmentation_stack[viewing_z]

        # Draw highlight
        extent = (offset.x, offset.x + self._image_slice_2d.shape[1],
                  offset.y + self._image_slice_2d.shape[0], offset.y)
        self._ax.imshow(segmentation_slice_at_viewing_z == label, cmap="gray", extent=extent, interpolation="nearest",
                        alpha=0.2)

    def _draw_drawing_shape(self):
        # Draw clicked positions
        if len(self._clicked_positions) > 0 and self._z == round(self._clicked_positions[0].z):
            x_positions = [position.x for position in self._clicked_positions]
            y_positions = [position.y for position in self._clicked_positions]
            self._ax.scatter(x_positions, y_positions, color="red", s=50, edgecolors="white", linewidths=1)
            if len(x_positions) > 1:
                self._ax.plot(x_positions, y_positions, color="white")

    def _on_mouse_double_click(self, event: MouseEvent):
        if self._display_settings.segmentation_channel is None:
            dialog.popup_error("No segmentation images selected",
                               "No segmentation images have been selected yet. Set a channel in the Edit menu.")
            return

        position = self._get_position_at(event.xdata, event.ydata)
        self._selected_position = position
        self._clicked_positions.clear()

        segmentation_stack, label = self._get_segmentation_stack_and_selected_label()
        if segmentation_stack is None or label is None:
            self._selected_position = None
            self.draw_view()
            self.update_status("Cannot select the cell here - no segmentation image available here.")
            return
        if label == 0:
            self.draw_view()
            self.update_status("Position doesn't have a segmentation mask yet. You can start drawing one using the"
                               "left-mouse button.")
            return

        self.draw_view()
        self.update_status(
            f"Double-clicked at {position}, which has label {label}.\nPress Shift+Delete to delete the mask."
            f"\nUse the left-mouse button to draw a selection, then use Insert or Enter to add that part to the"
            f"\nmask, or Delete or Backspace to remove that part from the mask.")

    def _on_mouse_single_click(self, event: MouseEvent):
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return  # Clicked outside the image

        is_right_click = event.button == 3
        is_left_click = event.button == 1
        if not is_right_click and not is_left_click:
            return  # Only left and right clicks are supported

        if self._selected_position is None or self._selected_position.time_point() != self._time_point:
            return  # No valid position selected

        if is_left_click:
            # Try to add a point
            if len(self._clicked_positions) > 0:
                if self._z != self._clicked_positions[0].z:
                    self._clicked_positions.clear()  # Start a new polygon if we change z or clicking button

            self._clicked_positions.append(Position(x, y, self._z, time_point=self._time_point))
            self.draw_view()
            self.update_status(
                "Added a point. Left-click somewhere else to draw another point, and press Enter or Insert to add the mask, or Delete or Backspace to remove the mask. Right-click somewhere to remove the last point.")
        else:
            # Try to remove a point
            if len(self._clicked_positions) > 0:
                if self._z != self._clicked_positions[0].z:
                    self._clicked_positions.clear()
                    self.draw_view()
                    self.update_status("Removed all points, as they were on a different Z-layer.")
                    return
                self._clicked_positions.pop()
                self.draw_view()
                self.update_status("Removed the last point.")

    def _on_key_press(self, event: KeyEvent):
        if event.key in {"enter", "insert"}:
            self._try_add_label()
        elif event.key in {"delete", "backspace"}:
            self._try_remove_label()
        elif event.key == "m":
            # Exit view immediately
            self._force_exit_to_tracking_editor()
        else:
            super()._on_key_press(event)

    def _check_valid_polygon(self) -> bool:
        """Checks if the current polygon is valid, and updates the status message if not. Returns True if the polygon is
        valid, False otherwise."""
        if len(self._clicked_positions) < 3:
            self.update_status("Need at least 3 points to draw a polygon. Use the left-mouse button to draw a polygon.")
            return False

        z_coord = int(self._clicked_positions[0].z)
        if z_coord != self._z:
            self.update_status(f"The polygon was drawn at z={z_coord}, so please move there first.")
            return False

        return True

    def _try_add_label(self):
        if not self._check_valid_polygon():
            return

        segmentation_image, label = self._get_segmentation_stack_and_selected_label()
        if segmentation_image is None or label is None:
            self.update_status("No segmentation image available here.")
            return

        # Need to add a new label if the selected position has no label
        if label == 0:
            label = _find_new_label(segmentation_image)

            # Check if that label will still fit the dtype
            try:
                if label > numpy.iinfo(segmentation_image.dtype).max:
                    bits = numpy.iinfo(segmentation_image.dtype).bits
                    self.update_status(f"We ran out of labels for this {bits}-bits image. Please delete some "
                                       f" first, or convert the image.")
                    return
            except ValueError:
                pass  # Float image, ignore

        x_min, y_min, z_coord, mask = self._to_mask(self._clicked_positions)

        # Create the action
        label_action = _SetLabelAction(segmentation_image, mask, x_min, y_min, z_coord, label, delete=False)

        # Perform the action
        self._clicked_positions.clear()
        self._perform_action(label_action)

    def _perform_action(self, action: UndoableAction):
        self._segmentation_image_undo_redo.do(action, self._experiment)
        self.draw_view()

    def _try_remove_label(self):
        if not self._check_valid_polygon():
            return

        segmentation_image, label = self._get_segmentation_stack_and_selected_label()
        if segmentation_image is None or label is None:
            self.update_status("No segmentation image available here.")
            return
        if label == 0:
            self.update_status("Currently selected position has no segmentation mask, so cannot delete anything.")
            return

        x_min, y_min, z_coord, mask = self._to_mask(self._clicked_positions)

        # Create the action
        label_action = _SetLabelAction(segmentation_image, mask, x_min, y_min, z_coord, label, delete=True)

        # Perform the action
        self._clicked_positions.clear()
        self._perform_action(label_action)

    def _to_mask(self, positions: List[Position]) -> Tuple[int, int, int, ndarray]:
        """Converts a polygon (consisting of a list of positions) to a mask. Returns the x, y, z coordinates of the mask
        (image coords) and the mask itself. Raises UserError if the mask selection is invalid."""
        if len(positions) < 3:
            raise UserError("Segmentation mask error", "Need at least 3 points to draw a mask.")

        # Build the mask
        x_coords = numpy.array([position.x for position in positions])
        y_coords = numpy.array([position.y for position in positions])
        x_min = int(x_coords.min())
        y_min = int(y_coords.min())
        x_coords -= x_min
        y_coords -= y_min
        width = math.ceil(x_coords.max() + 1)
        height = math.ceil(y_coords.max() + 1)
        image = PIL.Image.new('L', (width, height), 0)
        polygon = [(x, y) for x, y in zip(x_coords, y_coords)]
        PIL.ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)
        mask = numpy.array(image)

        # Move the mask to the image coordinates
        offset = self._experiment.images.offsets.of_time_point(self._time_point)
        z_coord = int(positions[0].z - offset.z)
        y_min -= int(offset.y)
        x_min -= int(offset.x)

        # Crop the mask to the image size
        if x_min < 0:
            if mask.shape[1] + x_min <= 0:
                raise UserError("Segmentation mask error", "Mask is completely outside the image")
            mask = mask[:, -x_min:]
            x_min = 0
        if y_min < 0:
            if mask.shape[0] + y_min <= 0:
                raise UserError("Segmentation mask error", "Mask is completely outside the image")
            mask = mask[-y_min:, :]
            y_min = 0
        if x_min + mask.shape[1] > self._segmentation_image_stack.shape[2]:
            if x_min >= self._segmentation_image_stack.shape[2]:
                raise UserError("Segmentation mask error", "Mask is completely outside the image")
            mask = mask[:, :self._segmentation_image_stack.shape[2] - x_min]
        if y_min + mask.shape[0] > self._segmentation_image_stack.shape[1]:
            if y_min >= self._segmentation_image_stack.shape[1]:
                raise UserError("Segmentation mask error", "Mask is completely outside the image")
            mask = mask[:self._segmentation_image_stack.shape[1] - y_min, :]

        return x_min, y_min, z_coord, mask

    def get_extra_menu_options(self) -> Dict[str, Any]:
        menu_options = {
            **super().get_extra_menu_options(),
            "Edit//Labeling-Delete current label [Shift+Delete]": self._delete_current_label,
        }

        image_loader = self._experiment.images.image_loader()
        segmentation_colormap = image_coloring.get_segmentation_colormap()

        for channel in image_loader.get_channels():
            channel_description = self._experiment.images.get_channel_description(channel)
            menu_label = "Channel " + channel_description.channel_name
            if channel == self._display_settings.segmentation_channel:
                menu_label += " (currently selected)"
            elif not image_loader.can_save_images(channel):
                menu_label += " (not writable)"
            elif channel_description.colormap.name != segmentation_colormap.name:
                menu_label += " (no segmentation)"
            else:
                menu_label += " (OK)"

            menu_options[f"Edit//Settings-Set segmentation image channel//" + menu_label] \
                = partial(self._set_segmentation_image_channel, channel)

        return menu_options

    def _delete_current_label(self):
        segmentation_stack, label = self._get_segmentation_stack_and_selected_label()
        if segmentation_stack is None or label is None:
            self.update_status("No label selected")
            return
        if label == 0:
            self.update_status("Currently selected position has no segmentation mask")
            return

        self._perform_action(_DeleteLabelAction(segmentation_stack, label))

    def _set_segmentation_image_channel(self, image_channel: ImageChannel):
        image_loader = self._experiment.images.image_loader()

        if not image_loader.can_save_images(image_channel):
            raise UserError("Channel not writable", "The selected channel is not writable. If you want to edit the"
                                                    " segmentation images, you need to save the images of the channel"
                                                    " as a folder of TIFF files.")
        description = self._experiment.images.get_channel_description(image_channel)

        # Check for segmentation colormap
        segmentation_colormap = image_coloring.get_segmentation_colormap()
        if description.colormap.name != segmentation_colormap.name:
            # Wrong colormap, ask user if they want to convert
            result = dialog.prompt_options("Not a segmentation channel",
                                           f"The selected channel (channel {description.channel_name}) is not a segmentation"
                                           " channel. What do you want to do?",
                                           option_1="Convert to segmentation channel")
            if result == 1:
                self._experiment.images.set_channel_description(image_channel,
                                                                description.with_colormap(segmentation_colormap))
                self._window.get_undo_redo().mark_unsaved_changes()
                self._display_settings.segmentation_channel = image_channel
                self._update_segmentation_image()
                self._window.redraw_all()
                self.update_status(
                    f"Converted channel {description.channel_name} to a segmentation channel.  You can now double-click a position to start editing its mask.")
            return

        # Already the right colormap, set as segmentation channel
        self._display_settings.segmentation_channel = image_channel
        self._update_segmentation_image()
        self._window.redraw_all()
        self.update_status(
            f"Set channel {description.channel_name} as segmentation channel. You can now double-click a position to start editing its mask.")

    def _has_unsaved_segmentation_changes(self) -> bool:
        """Returns True if there are unsaved changes in the segmentation image, False otherwise.

        We check the undo/redo queue of the current experiment. However, in case we just switched to another experiment,
        we would be looking at the wrong undo/redo queue. To be safe, we check for experiment switches, and if yes,
        we always return True.
        """
        return self._segmentation_image_undo_redo.has_unsaved_changes()
