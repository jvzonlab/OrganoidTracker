from typing import Optional, List, Tuple

import numpy
from numpy import ndarray


class LocationMap:
    """A map of (x,y) to an object. Cannot store values for negative x and y. Stores everything in a continuous array
     in memory, so if you store values at large (x,y) this object will consume a lot of memory."""

    _array: ndarray  # Indexed as y, x

    def __init__(self):
        self._array = numpy.full((4, 4), None, dtype=object)

    def set(self, x: int, y: int, value: Optional[object]):
        """Sets the value in the grid to be equal to the given object."""
        if x < 0 or y < 0:
            raise ValueError(f"x and y must be positive or 0, but were {x} and {y}")

        # Resize if necessary (we only double in size to avoid allocating new arrays all the time)
        if x >= self._array.shape[1] or y >= self._array.shape[0]:
            new_x_size = self._array.shape[1]
            while x >= new_x_size:
                new_x_size *= 2
            new_y_size = self._array.shape[0]
            while y >= new_y_size:
                new_y_size *= 2
            old_array = self._array
            self._array = numpy.full((new_y_size, new_x_size), None, dtype=object)
            self._array[0:old_array.shape[0], 0:old_array.shape[1]] = old_array

        self._array[y, x] = value

    def set_area(self, min_x: int, min_y: int, max_x: int, max_y: int, value: Optional[object]):
        """Sets the values in the given area of the grid all equal to the given value."""
        if min_x < 0 or min_y < 0:
            raise ValueError(f"x and y must be positive or 0, but were {x} and {y}")

        self.set(max_x, max_y, value)  # So that array is resized correctly

        self._array[min_y:max_y + 1, min_x:max_x + 1] = value

    def get_nearby(self, around_x: float, around_y: float) -> Optional[object]:
        """Gets the value at the given x and y position, searching for at most 2 indices next to it until a value that
        isn't None is returned."""
        # We need to work with integers in array access
        around_x = int(round(around_x))
        around_y = int(round(around_y))

        for dy in [0, -1, 1, -2, 2]:
            for dx in [0, -1, 1, -2, 2]:
                x = around_x + dx
                y = around_y + dy
                if x < 0 or y < 0 or x >= self._array.shape[1] or y >= self._array.shape[0]:
                    continue
                value = self._array[y, x]
                if value is not None:
                    return value
        return None

    def find_object(self, obj: object) -> Tuple[Optional[int], Optional[int]]:
        """Finds back the given object in the location map. If it occurs multiple
        times in the location map, then this method returns an arbitrary location.
        If the object doesn't occur in the map, then this method returns (None, None)."""
        y_indices, x_indices = numpy.where(self._array == obj)
        if len(x_indices) == 0:
            return None, None
        return int(x_indices[0]), int(y_indices[0])
