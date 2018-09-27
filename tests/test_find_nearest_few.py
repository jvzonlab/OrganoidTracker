import unittest

from autotrack.core import TimePoint
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.linking.find_nearest_neighbors import find_nearest_particles


class TestFindNearestFew(unittest.TestCase):

    def test_find_two(self):
        time_point = TimePoint(2)
        particles = ParticleCollection()
        particles.add(Particle(10,20,0).with_time_point(time_point))
        particles.add(Particle(11,20,0).with_time_point(time_point))
        particles.add(Particle(100,20,0).with_time_point(time_point))
        found = find_nearest_particles(particles, time_point, Particle(40,20,0), 1.1)
        self.assertEqual(2, len(found), "Expected to find two particles that are close to each other")

    def test_find_one(self):
        time_point = TimePoint(2)
        particles = ParticleCollection()
        particles.add(Particle(10, 20, 0).with_time_point(time_point))
        particles.add(Particle(11, 20, 0).with_time_point(time_point))
        particles.add(Particle(100, 20, 0).with_time_point(time_point))
        found = find_nearest_particles(particles, time_point, Particle(80, 20, 0), 1.1)
        self.assertEqual(1, len(found), "Expected to find one particle that is close enough")

    def test_zero_tolerance(self):
        time_point = TimePoint(2)
        particles = ParticleCollection()
        particles.add(Particle(10, 20, 0).with_time_point(time_point))
        particles.add(Particle(11, 20, 0).with_time_point(time_point))
        particles.add(Particle(100, 20, 0).with_time_point(time_point))
        found = find_nearest_particles(particles, time_point, Particle(40,20,0), 1)
        self.assertEqual(1, len(found), "Tolerance is set to 1.0, so only one particle may be found")