import unittest

from ai_track.comparison.report import ComparisonReport, Category
from ai_track.core.position import Position


class TestFindNearestFew(unittest.TestCase):

    def test_basic(self):
        report = ComparisonReport()
        report.title = "My title"
        report.summary = "My summary"
        report.add_data(Category("My category"), Position(0, 0, 0, time_point_number=1), "This is a test")
        report.add_data(Category("My category"), Position(0, 0, 0, time_point_number=2), "This is another test")
        report.add_data(Category("Other category"), Position(0, 0, 0, time_point_number=3), "Last test")
        self.assertEquals(f"""My title
--------

My summary

My category: (2)
* {Position(0, 0, 0, time_point_number=1)} - This is a test
* {Position(0, 0, 0, time_point_number=2)} - This is another test

Other category: (1)
* {Position(0, 0, 0, time_point_number=3)} - Last test
""", str(report))
