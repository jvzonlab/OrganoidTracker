# File originally written by Guizela Huelsz Prince


class Axis:
    def __init__(self, x, t):
        self.x = x
        self.t = t

    def add_point(self, x):
        self.x = self.x + [x]

    # delete last point
    def delete_point(self):
        self.x = self.x[:-1]


def add_axis_to_axes_list(x, t, axes_list):
    axes_list.append(Axis(x, t))
