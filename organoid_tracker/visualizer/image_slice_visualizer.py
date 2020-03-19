from typing import Dict, Any, Type, Optional

import numpy
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent

from organoid_tracker.core import TimePoint
from organoid_tracker.gui.window import Window
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS
from organoid_tracker.visualizer import Visualizer
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer
from organoid_tracker.visualizer.standard_image_visualizer import StandardImageVisualizer


class ImageSliceViewer(ExitableImageVisualizer):
    """Showing image slices. Double-click somewhere to move the slice to that point."""

    _x: int = 200
    _y: int = 200
    _right_axes: Axes
    _bottom_axes: Axes
    _first_draw: bool = True

    _time_point_images: Optional[ndarray] = None # Full 3d-image of time point

    def __init__(self, window: Window, parent_viewer: Type[Visualizer] = StandardImageVisualizer):
        super().__init__(window, parent_viewer)
        self._right_axes = self._axes[1]
        self._bottom_axes = self._axes[2]
        self._axes[3].set_visible(False)

    def _get_subplots_config(self) -> Dict[str, Any]:
        return {
            "nrows": 2,
            "ncols": 2,
            "sharex": "col",
            "sharey": "row",
            "gridspec_kw":  {"width_ratios": [3, 1], "height_ratios": [3, 1]}
        }

    def _load_2d_image(self):
        # Disabled, as we need to work with 3d images here
        self._image_slice_2d = None

    def _calculate_time_point_metadata(self):
        # We need to load the full 3D image
        self._time_point_images = self._experiment.images.get_image_stack(self._time_point,
                                                                          self._display_settings.image_channel)
        super()._calculate_time_point_metadata()

    def _draw_image(self):
        if self._time_point_images is None:
            return

        offset = self._experiment.images.offsets.of_time_point(self._time_point)

        # Right: draw slice in yz
        self._right_axes.set_facecolor(self._ax.get_facecolor())
        image = numpy.flip(numpy.moveaxis(self._time_point_images[:, :, int(self._x - offset.x)], 0, 1), 0)
        extent = (offset.z, offset.z + image.shape[1], offset.y, offset.y + image.shape[0])
        self._right_axes.imshow(image, cmap=self._color_map, extent=extent, aspect="auto")
        self._right_axes.axvline(self._z, color=SANDER_APPROVED_COLORS[2])
        if self._right_axes.xaxis_inverted():
            self._right_axes.invert_xaxis()

        # Bottom: draw slice in xz
        self._bottom_axes.set_facecolor(self._ax.get_facecolor())
        image = self._time_point_images[:, int(self._y - offset.y), :]
        extent = (offset.x, offset.x + image.shape[1], offset.z + image.shape[0], offset.z)
        self._bottom_axes.imshow(image, cmap=self._color_map, extent=extent, aspect="auto")
        self._bottom_axes.axhline(self._z, color=SANDER_APPROVED_COLORS[1])
        if self._bottom_axes.yaxis_inverted():
            self._bottom_axes.invert_yaxis()

        # Draw main image
        main_axes = self._ax
        main_image = self._time_point_images[int(self._z - offset.z)]
        extent = (offset.x, offset.x + main_image.shape[1], offset.y + main_image.shape[0], offset.y)
        main_axes.imshow(main_image, cmap=self._color_map, extent=extent, aspect="auto")
        main_axes.axhline(self._y, color=SANDER_APPROVED_COLORS[1])
        main_axes.axvline(self._x, color=SANDER_APPROVED_COLORS[2])

        # If we're new, set zoom to whole z view
        if self._has_default_axes_limit(self._bottom_axes):
            if not self._ax.yaxis_inverted():
                self._ax.invert_yaxis()
            self._right_axes.set_xlim(offset.z, offset.z + self._time_point_images.shape[0])
            self._bottom_axes.set_ylim(offset.z, offset.z + self._time_point_images.shape[0])

    def _on_mouse_click(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None or not event.dblclick:
            super()._on_mouse_click(event)
            return

        if event.inaxes == self._ax:
            self._x = int(event.xdata)
            self._y = int(event.ydata)
            self.draw_view()
        elif event.inaxes == self._bottom_axes:
            self._x = int(event.xdata)
            self._move_to_z(int(event.ydata))
        elif event.inaxes == self._right_axes:
            self._y = int(event.ydata)
            self._move_to_z(int(event.xdata))
        else:
            super()._on_mouse_click(event)

    def _has_default_axes_limit(self, axes: Axes) -> bool:
        """If one of the axis has a length equal or smaller than 1, it's just the default value."""
        xlim = axes.get_xlim()
        if abs(xlim[1] - xlim[0]) <= 1:
            return True

        ylim = axes.get_ylim()
        if abs(ylim[1] - ylim[0]) <= 1:
            return True

        return False
