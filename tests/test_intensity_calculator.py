import unittest

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.position_analysis import intensity_calculator


class TestIntensityCalculator(unittest.TestCase):


    def test_normalization(self):
        position_1 = Position(1, 0, 0, time_point_number=0)
        position_2 = Position(2, 0, 0, time_point_number=0)
        position_3 = Position(3, 0, 0, time_point_number=0)

        experiment = Experiment()
        position_data = experiment.position_data
        position_data.set_position_data(position_1, "intensity", 8)
        position_data.set_position_data(position_2, "intensity", 10)
        position_data.set_position_data(position_3, "intensity", 12)
        position_data.set_position_data(position_1, "intensity_volume", 100)
        position_data.set_position_data(position_2, "intensity_volume", 100)
        position_data.set_position_data(position_3, "intensity_volume", 200)

        intensity_calculator.perform_intensity_normalization(experiment)
        intensity1 = intensity_calculator.get_normalized_intensity(experiment, position_1)
        intensity2 = intensity_calculator.get_normalized_intensity(experiment, position_2)
        intensity3 = intensity_calculator.get_normalized_intensity(experiment, position_3)

        # check if median is indeed 100
        self.assertAlmostEqual(100, sorted([intensity1, intensity2,intensity3])[1], delta=0.0001)

        # check if lowest is indeed 0 (background correction)
        self.assertAlmostEqual(0, intensity1, delta=0.0001)
