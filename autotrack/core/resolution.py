from typing import Tuple


class ImageResolution:
    """Represents the resolution of a 3D image."""
    pixel_size_zyx_um: Tuple[float, float, float]
    time_point_interval_m: float  # Time between time points in minutes

    def __init__(self, pixel_size_x_um: float, pixel_size_y_um: float, pixel_size_z_um: float, time_point_interval_m: float):
        self.pixel_size_zyx_um = (pixel_size_z_um, pixel_size_y_um, pixel_size_x_um)
        self.time_point_interval_m = time_point_interval_m

    @property
    def time_point_interval_h(self) -> float:
        return self.time_point_interval_m / 60
