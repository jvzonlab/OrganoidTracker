from typing import Optional, List, Tuple

import numpy
from numpy import ndarray


class LocationMap:
    """A map of (x,y) to an object. Cannot store values for negative x and y. Stores everything in a continuous array
     in memory, so if you store values at large (x,y) this object will consume a lot of memory. You can set a large
    cell size to combat this effect."""

    _array: ndarray  # Indexed as y, x
    _cell_size_x: float
    _cell_size_y: float

    def __init__(self, *, cell_size_x: float = 1, cell_size_y: float = 1):
        """Initializes the location map. You can set a different cell size. A cell size of 2 means that objects stored
        at (0, 0) and (1, 1) will overwrite each other, as they would be stored in the same cell."""
        self._array = numpy.full((4, 4), None, dtype=object)
        self._cell_size_x = cell_size_x
        self._cell_size_y = cell_size_y

    def set(self, x: float, y: float, value: Optional[object]):
        """Sets the value in the grid to be equal to the given object."""
        if x < 0 or y < 0:
            raise ValueError(f"x and y must be positive or 0, but were {x_array} and {y_array}")

        x_array = int(x / self._cell_size_x)
        y_array = int(y / self._cell_size_y)

        # Resize if necessary (we only double in size to avoid allocating new arrays all the time)
        if x_array >= self._array.shape[1] or y_array >= self._array.shape[0]:
            new_x_size = self._array.shape[1]
            while x_array >= new_x_size:
                new_x_size *= 2
            new_y_size = self._array.shape[0]
            while y_array >= new_y_size:
                new_y_size *= 2
            old_array = self._array
            self._array = numpy.full((new_y_size, new_x_size), None, dtype=object)
            self._array[0:old_array.shape[0], 0:old_array.shape[1]] = old_array

        self._array[y_array, x_array] = value

    def set_area(self, min_x: float, min_y: float, max_x: float, max_y: float, value: Optional[object]):
        """Sets the values in the given area of the grid all equal to the given value."""
        if min_x < 0 or min_y < 0:
            raise ValueError(f"x and y must be positive or 0, but were {x} and {y}")

        self.set(max_x, max_y, value)  # So that array is resized correctly

        min_array_x = int(min_x / self._cell_size_x)
        min_array_y = int(min_y / self._cell_size_y)
        max_array_x = int(max_x / self._cell_size_x)
        max_array_y = int(max_y / self._cell_size_y)
        self._array[min_array_y:max_array_y + 1, min_array_x:max_array_x + 1] = value

    def get_nearby(self, around_x: float, around_y: float) -> Optional[object]:
        """Gets the value at the given x and y position, searching for at most 2 indices next to it until a value that
        isn't None is returned."""
        for dy in [0, -1, 1, -2, 2]:
            for dx in [0, -1, 1, -2, 2]:
                x = around_x + dx
                y = around_y + dy
                if x < 0 or y < 0 or x >= self._array.shape[1] or y >= self._array.shape[0]:
                    continue
                value = self._array[int(round(y / self._cell_size_y)), int(round(x / self._cell_size_x))]
                if value is not None:
                    return value
        return None

    def find_object(self, obj: object) -> Tuple[Optional[float], Optional[float]]:
        """Finds back the given object in the location map. If it occurs multiple
        times in the location map, then this method returns an arbitrary location.
        If the object doesn't occur in the map, then this method returns (None, None)."""
        y_indices, x_indices = numpy.where(self._array == obj)
        if len(x_indices) == 0:
            return None, None
        return float(x_indices[0] * self._cell_size_x), float(y_indices[0] * self._cell_size_y)
