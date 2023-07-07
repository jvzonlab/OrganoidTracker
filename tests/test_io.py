import math
import os
from unittest import TestCase

import numpy

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io
from tempfile import TemporaryDirectory

class TestIO(TestCase):

    def test_numpy(self):
        """Ensure numpy floats can be saved and loaded. (Not possible by default in most json serializers.)"""
        position = Position(1, 1, 1, time_point_number=1)

        experiment = Experiment()
        experiment.positions.add(position)
        experiment.position_data.set_position_data(position, "test_key", numpy.sqrt(5))

        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            io.save_data_to_json(experiment, file)

            experiment = io.load_data_file(file)
            self.assertAlmostEqual(math.sqrt(5), experiment.position_data.get_position_data(position, "test_key"))
