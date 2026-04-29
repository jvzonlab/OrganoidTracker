from unittest import TestCase

from organoid_tracker.util.moving_average import LinesAverage


class TestLinesAverage(TestCase):


    def test_non_interpolating(self):
        """Simple case: both lines have an x time resolution of 1.5."""
        line_a = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], [3,  4,  5,  6, 7, 8,  9, 10, 11]
        line_b = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], [-2, -3, -1, 4, 5, -2, -10, 3, 2]

        lines_average = LinesAverage(line_a, line_b, x_step_size=1.5)
        self.assertEqual(1.5, lines_average.x_step_size)
        self.assertListEqual([0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], lines_average.x_values.tolist())
        self.assertListEqual([0.5, 0.5, 2, 5, 6, 3, -0.5, 6.5, 6.5], lines_average.mean_values.tolist())

    def test_interpolating(self):
        """Here, two lines have different time resolutions: 0.4 vs 1. Yet, we sample at a resolution of 1, so we need
        to interpolate for one point (at x=1) in line_highres."""
        line_highres = [0, 0.4, 0.8, 1.2, 1.6, 2.0], [0, 0.5, 1, 1.5, 2, 2.5]
        line_lowres = [0, 1, 2], [2, 4, 6]

        lines_average = LinesAverage(line_highres, line_lowres, x_step_size=1)

        self.assertEqual(1, lines_average.x_step_size)
        self.assertListEqual([0, 1, 2], lines_average.x_values.tolist())
        self.assertListEqual([1, 2.625, 4.25], lines_average.mean_values.tolist())
