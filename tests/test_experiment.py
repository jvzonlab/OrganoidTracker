from unittest import TestCase

import numpy

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.image_loading.array_image_loader import SingleImageLoader


class TestExperiment(TestCase):


    def test_clear_tracking_data(self):
        experiment = Experiment()

        position_a = Position(0, 0, 0, time_point_number=0)
        position_b = Position(0, 0, 0, time_point_number=1)

        # Add some positions and a link
        experiment.positions.add(position_a)
        experiment.positions.add(position_b)
        experiment.links.add_link(position_a, position_b)
        experiment.global_data.set_data("test", "value")

        image_loader = SingleImageLoader(numpy.ones((10, 10, 10)))
        experiment.images.image_loader(image_loader)
        experiment.images.set_resolution(ImageResolution(0.3, 0.3, 1.0, 12))

        # Clear tracking data, and check that everything is gone
        experiment.clear_tracking_data()
        self.assertEqual(len(experiment.positions), 0)
        self.assertEqual(len(experiment.links), 0)
        self.assertIsNone(experiment.global_data.get_data("test"))

        # But the image loader and resolution should still be there
        self.assertEqual(image_loader, experiment.images.image_loader())
        self.assertEqual(0.3, experiment.images.resolution().pixel_size_x_um)
