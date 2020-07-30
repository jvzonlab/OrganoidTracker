from typing import Tuple


class ImageResolution:
    """Represents the resolution of a 3D image. X and y resolution must be equal. The fields in this class should be
    treated as immutable: don't modify their values after creation."""

    # If you want to define 1 um = 1 px, use this resolution. Useful if you want ot measure distances in pixels instead
    # of micrometers.
    PIXELS: "ImageResolution" = ...  # Initialized after definition of class.

    pixel_size_zyx_um: Tuple[float, float, float]
    time_point_interval_m: float  # Time between time points in minutes

    def __init__(self, pixel_size_x_um: float, pixel_size_y_um: float, pixel_size_z_um: float, time_point_interval_m: float):
        if pixel_size_x_um < 0 or pixel_size_z_um < 0 or time_point_interval_m < 0:
            raise ValueError("Resolution cannot be negative")

        self.pixel_size_zyx_um = (pixel_size_z_um, pixel_size_y_um, pixel_size_x_um)
        self.time_point_interval_m = time_point_interval_m

    @property
    def time_point_interval_h(self) -> float:
        return self.time_point_interval_m / 60

    @property
    def pixel_size_x_um(self) -> float:
        return self.pixel_size_zyx_um[2]

    @property
    def pixel_size_y_um(self) -> float:
        return self.pixel_size_zyx_um[1]

    @property
    def pixel_size_z_um(self) -> float:
        return self.pixel_size_zyx_um[0]

    def __repr__(self) -> str:
        return f"ImageResolution({self.pixel_size_x_um}, {self.pixel_size_y_um}, {self.pixel_size_z_um}," \
               f" {self.time_point_interval_m})"

# See typehint at beginning of ImageResolution class
ImageResolution.PIXELS = ImageResolution(1, 1, 1, 1)
