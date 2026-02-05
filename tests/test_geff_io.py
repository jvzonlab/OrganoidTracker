import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import geff_io


class TestGeffIO(TestCase):


    def test_geff_loading(self):
        # We load a sample GEFF file to ensure no exceptions are raised
        script_file_location = os.path.dirname(__file__)
        file_name = os.path.join(script_file_location, "resources", "Fluo-N3DH-CE 01_GT.geff")
        experiment = geff_io.load_data_file(file_name)

        position_count = len(experiment.positions)
        self.assertEqual(23802, position_count)

        time_point_count = len(experiment.positions.time_points())
        self.assertEqual(195, time_point_count)

    def test_geff_saving(self):

        # Build a sample experiment
        experiment = Experiment()
        pos_a = Position(10.0, 20.0, 30.0, time_point_number=1)
        pos_b = Position(40.0, 50.0, 60.0, time_point_number=2)
        pos_c = Position(70.0, 80.0, 90.0, time_point_number=3)
        pos_x = Position(15.0, 25.0, 35.0, time_point_number=1)
        experiment.positions.add(pos_a)
        experiment.positions.add(pos_b)
        experiment.positions.add(pos_c)
        experiment.positions.add(pos_x)
        experiment.links.add_link(pos_a, pos_b)
        experiment.links.add_link(pos_b, pos_c)
        experiment.positions.set_position_data(pos_a, "cell_type", "FOO")
        experiment.positions.set_position_data(pos_b, "cell_type", "FOO")
        experiment.positions.set_position_data(pos_x, "cell_type", "BAR")
        experiment.positions.set_position_data(pos_x, "some_intensity", 3.4)
        experiment.positions.set_position_data(pos_x, "some_list", [4.5, 0.3])
        experiment.links.set_link_data(pos_a, pos_b, "some_link_data", True)

        # Save and reload
        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test.geff")
            geff_io.save_data_file(experiment, file)
            experiment_reloaded = geff_io.load_data_file(file)

        self.assertEqual(len(experiment.positions), len(experiment_reloaded.positions))