import unittest

from autotrack.comparison.report import ComparisonReport, Category
from autotrack.core.particles import Particle


class TestFindNearestFew(unittest.TestCase):

    def test_basic(self):
        report = ComparisonReport()
        report.title = "My title"
        report.summary = "My summary"
        report.add_data(Category("My category"), Particle(0, 0, 0).with_time_point_number(1), "This is a test")
        report.add_data(Category("My category"), Particle(0, 0, 0).with_time_point_number(2), "This is another test")
        report.add_data(Category("Other category"), Particle(0, 0, 0).with_time_point_number(3), "Last test")
        self.assertEquals(f"""My title
--------

My summary

My category: (2)
* {Particle(0, 0, 0).with_time_point_number(1)} - This is a test
* {Particle(0, 0, 0).with_time_point_number(2)} - This is another test

Other category: (1)
* {Particle(0, 0, 0).with_time_point_number(3)} - Last test
""", str(report))
