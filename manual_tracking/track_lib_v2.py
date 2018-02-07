# Old version of the track lib by Guizela, necessary to load tracks
import numpy as np


class Track:
    def __init__(self, x, t):
        self.x = np.array([x])
        self.t = np.array([t])

    def get_pos(self, t):
        q = np.where(self.t == t)[0]
        if len(q) > 0:
            ind = q[0]
            return self.x[ind]
        else:
            return np.array([])

    def add_point(self, x, t):
        # check if time point t is already there
        q = np.where(self.t == t)[0]
        if len(q) == 0:
            # if not, then add point
            self.x = np.vstack((self.x, x))
            self.t = np.append(self.t, t)
            # then sort t and x so that all points are sorted in time
            ind_sort = np.argsort(self.t)
            self.t = self.t[ind_sort]
            self.x = self.x[ind_sort]
        else:
            # replace point
            ind = q[0]
            print(t)
            self.x[ind, :] = x
            self.t[ind] = t

    def delete_point(self, t):
        # find index of point with time t
        q = [i for i, j in enumerate(self.t) if j == t]
        # if it exists
        if q:
            ind = q[0]
            # remove time
            self.t = np.delete(self.t, ind)
            # and position
            self.x = np.delete(self.x, ind, axis=0)
