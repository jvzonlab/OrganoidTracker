import unittest

import core
from core import Particle


class TestFindNearestFew(unittest.TestCase):

    def find_three_particles(self):
        system = [Particle(0,0,0), Particle(0,7,0), Particle(0,2,0), Particle(0,1,0), Particle(0,3,0)]
        self.assertEquals(
            set(Particle(0,0,0), Particle(0,2,0), Particle(0,1,0)),
            core.get_closest_n_particles(system, Particle(0, -1, 0), 3))