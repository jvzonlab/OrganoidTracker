import unittest

from imaging import Particle
from particle_detection.detector_for_experiment import Peak


class TestPeakMerge(unittest.TestCase):

    def test_split_in_two_parts(self):
        # Construct peak structure
        peak_1, peak_2, peak_3, peak_4 = Peak(0, 0, 1), Peak(0, 0, 2), Peak(0, 0, 3), Peak(0, 0, 4)
        peak_1.above = peak_2
        peak_2.above = peak_3
        peak_3.above = peak_4

        self.assertEquals([Particle(0, 0, 1.5), Particle(0, 0, 3.5)],
                          peak_1.to_particle(max_cell_height=2))

    def test_split_in_four_parts(self):
        # Construct peak structure
        peak_1, peak_2, peak_3, peak_4 = Peak(0, 0, 1), Peak(0, 0, 2), Peak(0, 0, 3), Peak(0, 0, 4)
        peak_1.above = peak_2
        peak_2.above = peak_3
        peak_3.above = peak_4

        self.assertEquals([Particle(0, 0, 1), Particle(0, 0, 2), Particle(0, 0, 3), Particle(0, 0, 4)],
                          peak_1.to_particle(max_cell_height=1))

    def test_split_in_one_part(self):
        # Construct peak structure
        peak_1, peak_2, peak_3, peak_4 = Peak(0, 0, 1), Peak(0, 0, 2), Peak(0, 0, 3), Peak(0, 0, 4)
        peak_1.above = peak_2
        peak_2.above = peak_3
        peak_3.above = peak_4

        self.assertEquals([Particle(0, 0, 2.5)], peak_1.to_particle(max_cell_height=4))
