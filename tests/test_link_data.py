from unittest import TestCase

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position


class TestLinkData(TestCase):
    """Tests that the deprecated link_data API still works, and writes through to the new links API."""

    # noinspection PyDeprecation
    def test_write_through(self):
        experiment = Experiment()
        position_1 = Position(1, 2, 3, time_point_number=4)
        position_2 = Position(4, 5, 6, time_point_number=5)
        experiment.links.add_link(position_1, position_2)

        # Test setting using the new API, retrieving using the old API
        experiment.links.set_link_data(position_1, position_2, "via_new_api", "test_value")
        self.assertEqual("test_value", experiment.link_data.get_link_data(position_1, position_2, "via_new_api"))

        # Test setting using the old API, retrieving using the new API
        experiment.link_data.set_link_data(position_1, position_2, "via_old_api", "test_value_2")
        self.assertEqual("test_value_2", experiment.links.get_link_data(position_1, position_2, "via_old_api"))
