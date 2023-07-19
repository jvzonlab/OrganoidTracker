from typing import Tuple

from numpy import ndarray


class BoundingBox:
    """Bounding box. Min is inclusive, max is exclusive. Fields should be left immutable."""
    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int

    def __init__(self, min_x: int, min_y: int, min_z: int, max_x: int, max_y: int, max_z: int):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        if min_x > max_x:
            raise ValueError(f"min_x > max_x: {min_x} > {max_x}")
        if min_y > max_y:
            raise ValueError(f"min_y > max_y: {min_y} > {max_z}")
        if min_z > max_z:
            raise ValueError(f"min_z > max_z: {min_y} > {max_z}")

    def expanded(self, x: int = 0, y: int = 0, z: int = 0) -> "BoundingBox":
        """Expands the bounding box with the given number of pixels on all directions."""
        if x < 0 or y < 0 or z < 0:
            raise ValueError(f"x, y and z must all be non-negative, but were {x}, {y} and {z}")
        return BoundingBox(self.min_x - x, self.min_y - y, self.min_z - z,
                           self.max_x + x, self.max_y + y, self.max_z + z)

    def __repr__(self):
        return f"BoundingBox({self.min_x}, {self.min_y}, {self.min_z}, {self.max_x}, {self.max_y}, {self.max_z})"

    def get_size_zyx(self) -> Tuple[int, int, int]:
        """Gets the size of this bounding box."""
        return self.max_z - self.min_z, self.max_y - self.min_y, self.max_x - self.min_x


# A bounding box around (0, 0, 0) with a size of 1.
ONE = BoundingBox(0, 0, 0, 1, 1, 1)
