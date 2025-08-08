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
        positions = experiment.positions
        positions.add(position_1)
        positions.add(position_2)
        positions.add(position_3)

        positions.set_position_data(position_1, "intensity", 8)
        positions.set_position_data(position_2, "intensity", 10)
        positions.set_position_data(position_3, "intensity", 12)
        positions.set_position_data(position_1, "intensity_volume", 100)
        positions.set_position_data(position_2, "intensity_volume", 100)
        positions.set_position_data(position_3, "intensity_volume", 200)

        intensity_calculator.perform_intensity_normalization(experiment)
        intensity1 = intensity_calculator.get_normalized_intensity(experiment, position_1)
        intensity2 = intensity_calculator.get_normalized_intensity(experiment, position_2)
        intensity3 = intensity_calculator.get_normalized_intensity(experiment, position_3)

        print(intensity1, intensity2, intensity3)

        # check if median is indeed 1
        self.assertAlmostEqual(1, sorted([intensity1, intensity2,intensity3])[1], delta=0.0001)

        # check if lowest per volume is indeed 0 (background correction)
        self.assertAlmostEqual(0, intensity3, delta=0.0001)
