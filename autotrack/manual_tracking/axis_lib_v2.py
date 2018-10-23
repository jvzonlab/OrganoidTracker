# File originally written by Guizela Huelsz Prince

import pickle


class Axis(object):
    def __init__(self, x, t):
        self.x = x
        self.t = t

    def add_point(self, x):
        self.x = self.x + [x]

    # delete last point
    def delete_point(self):
        if self.x:
            self.x = self.x[:-1]


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
