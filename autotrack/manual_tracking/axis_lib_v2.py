# File originally written by Guizela Huelsz Prince

import pickle
from typing import List, Optional

import numpy
from numpy import ndarray
from scipy import interpolate


class Axis(object):
    t: int
    x: List[List[float]]
    interpolated: Optional[List[ndarray]]

    def __init__(self, x, t):
        self.x = x
        self.t = t

    def add_point(self, x):
        self.x = self.x + [x]

    # delete last point
    def delete_point(self):
        if self.x:
            self.x = self.x[:-1]

    def interpolate(self):
        """Adds an interpolation of 1001 points."""
        x = [row[0] for row in self.x]
        y = [row[1] for row in self.x]
        z = [row[2] for row in self.x]

        l = len(self.x)
        if l > 1:
            if l > 3:
                k = 3
            else:  # so l in [2, 3]
                k = 1
            # noinspection PyTupleAssignmentBalance
            tck, u = interpolate.splprep([x, y], k=k)
            unew = numpy.arange(0, 1.001, 0.001)
            out = interpolate.splev(unew, tck)
            z = numpy.asarray([z[0] for i in out[0]])
            self.interpolated = [out[0], out[1], z]
        else:
            self.interpolated = [numpy.asarray(x), numpy.asarray(y), numpy.asarray(z)]

    def __str__(self) -> str:
        is_interpolated = hasattr(self, "interpolated")
        return f"Axis of length {len(self.x)}, interpolated:{is_interpolated}"


def add_axis_to_axes_list(x, t, axes_list):
    axes_list.append(Axis(x, t))


def remove_axis_from_axes_list(idx, axes_list):
    del axes_list[idx]


def save_axes_list(axes_list, path):
    # remove empty axes
    axes_list = [x for x in axes_list if x.x]

    pickle.dump(axes_list, open(path + 'crypt_axes.p', 'wb'))
    print('Saved axes.')


def load_axes_list(path):
    axes_list = pickle.load(open(path + 'crypt_axes.p', 'rb'), encoding='latin1')
    return axes_list
