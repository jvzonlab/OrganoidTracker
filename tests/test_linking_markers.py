import unittest

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.errors import Error


class TestLinkingMarkers(unittest.TestCase):

    def test_error_marker(self):
        particle_links = ParticleLinks()

        self.assertEquals(None, linking_markers.get_error_marker(particle_links, Particle(0, 0, 0).with_time_point_number(0)),
                          "non-existing particle must have no error marker")

        particle = Particle(2, 2, 2).with_time_point_number(2)
        self.assertEquals(None, linking_markers.get_error_marker(particle_links, particle), "no error marker was set")

        linking_markers.set_error_marker(particle_links, particle, Error.MOVED_TOO_FAST)
        self.assertEquals(Error.MOVED_TOO_FAST, linking_markers.get_error_marker(particle_links, particle))
        self.assertFalse(linking_markers.is_error_suppressed(particle_links, particle, Error.MOVED_TOO_FAST))

        linking_markers.suppress_error_marker(particle_links, particle, Error.MOVED_TOO_FAST)
        self.assertEquals(None, linking_markers.get_error_marker(particle_links, particle), "error must be suppressed")
        self.assertTrue(linking_markers.is_error_suppressed(particle_links, particle, Error.MOVED_TOO_FAST))
