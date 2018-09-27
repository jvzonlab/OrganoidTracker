import unittest

from autotrack.core.particles import Particle, get_closest_n_particles


class TestFindNearestFew(unittest.TestCase):

    def test_find_three_particles(self):
        system = [Particle(0,0,0), Particle(0,7,0), Particle(0,2,0), Particle(0,1,0), Particle(0,3,0)]
        self.assertEquals(
            {Particle(0,0,0), Particle(0,2,0), Particle(0,1,0)},
            get_closest_n_particles(system, Particle(0, -1, 0), 3))