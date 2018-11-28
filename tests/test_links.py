import unittest

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle


class TestLinks(unittest.TestCase):

    def test_data(self):
        particle = Particle(0, 0, 0).with_time_point_number(0)
        links = ParticleLinks()
        links.set_particle_data(particle, "name", "AA")

        self.assertEquals("AA", links.get_particle_data(particle, "name"))

    def test_futures(self):
        particle = Particle(0, 0, 0).with_time_point_number(0)
        future_particle = Particle(1, 0, 0).with_time_point_number(1)
        links = ParticleLinks()
        links.add_link(particle, future_particle)

        self.assertEquals({future_particle}, links.find_futures(particle))
        self.assertEquals(set(), links.find_futures(future_particle))

    def test_pasts(self):
        particle = Particle(0, 0, 0).with_time_point_number(1)
        past_particle = Particle(1, 0, 0).with_time_point_number(0)
        links = ParticleLinks()
        links.add_link(particle, past_particle)

        self.assertEquals({past_particle}, links.find_pasts(particle))
        self.assertEquals(set(), links.find_pasts(past_particle))
