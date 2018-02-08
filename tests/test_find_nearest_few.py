import unittest
from imaging.positions import Frame, Particle
from nearest_neighbor_linking.find_nearest_few import find_nearest_particles


class TestFindNearestFew(unittest.TestCase):

    def test_find_two(self):
        frame = Frame(2, [Particle(10,20,0), Particle(11,20,0), Particle(100,20,0)])
        found = find_nearest_particles(frame, Particle(40,20,0), 1.1)
        self.assertEqual(2, len(found), "Expected to find two particles that are close to each other")

    def test_find_one(self):
        frame = Frame(2, [Particle(10, 20, 0), Particle(11, 20, 0), Particle(100, 20, 0)])
        found = find_nearest_particles(frame, Particle(80, 20, 0), 1.1)
        self.assertEqual(1, len(found), "Expected to find one particle that is close enough")

    def test_zero_tolerance(self):
        frame = Frame(2, [Particle(10,20,0), Particle(11,20,0), Particle(100,20,0)])
        found = find_nearest_particles(frame, Particle(40,20,0), 1)
        self.assertEqual(1, len(found), "Tolerance is set to 1.0, so only one particle may be found")