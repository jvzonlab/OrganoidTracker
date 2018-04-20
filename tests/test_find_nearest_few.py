import unittest
from imaging import TimePoint, Particle
from linking.find_nearest_neighbors import find_nearest_particles


class TestFindNearestFew(unittest.TestCase):

    def test_find_two(self):
        time_point = TimePoint(2)
        time_point.add_particle(Particle(10,20,0))
        time_point.add_particle(Particle(11,20,0))
        time_point.add_particle(Particle(100,20,0))
        found = find_nearest_particles(time_point, Particle(40,20,0), 1.1)
        self.assertEqual(2, len(found), "Expected to find two particles that are close to each other")

    def test_find_one(self):
        time_point = TimePoint(2)
        time_point.add_particle(Particle(10, 20, 0))
        time_point.add_particle(Particle(11, 20, 0))
        time_point.add_particle(Particle(100, 20, 0))
        found = find_nearest_particles(time_point, Particle(80, 20, 0), 1.1)
        self.assertEqual(1, len(found), "Expected to find one particle that is close enough")

    def test_zero_tolerance(self):
        time_point = TimePoint(2)
        time_point.add_particle(Particle(10, 20, 0))
        time_point.add_particle(Particle(11, 20, 0))
        time_point.add_particle(Particle(100, 20, 0))
        found = find_nearest_particles(time_point, Particle(40,20,0), 1)
        self.assertEqual(1, len(found), "Tolerance is set to 1.0, so only one particle may be found")