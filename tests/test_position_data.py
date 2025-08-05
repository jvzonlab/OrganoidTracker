import unittest

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position


class TestPositionData(unittest.TestCase):
    """Tests that the deprecated position_data API still works, and writes through to the new positions API."""

    def test_write_through(self):
        experiment = Experiment()
        position = Position(1, 2, 3, time_point_number=4)

        # Test setting using the new API, retrieving using the old API
        experiment.positions.set_position_data(position, "via_new_api", "test_value")
        self.assertEqual("test_value", experiment.position_data.get_position_data(position, "via_new_api"))

        # Test setting using the old API, retrieving using the new API
        experiment.position_data.set_position_data(position, "via_old_api", "test_value_2")
        self.assertEqual("test_value_2", experiment.positions.get_position_data(position, "via_old_api"))