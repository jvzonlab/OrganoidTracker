import json
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
        experiment.positions.set_position_data(position, "test_key", numpy.sqrt(5))

        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            io.save_data_to_json(experiment, file)

            experiment = io.load_data_file(file)
            self.assertAlmostEqual(math.sqrt(5), experiment.positions.get_position_data(position, "test_key"))

    def test_save_format(self):
        expected_data = {'version': 'v2',
                         'positions': [
                             {'time_point': 1, 'coords_xyz_px': [[1.0, 2.0, 3.0]]},
                             {'time_point': 2, 'coords_xyz_px': [[4.0, 5.0, 6.0]]}
                         ],
                         'tracks': [
                             {'time_point_start': 1, 'coords_xyz_px': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
                         ]}

        experiment = Experiment()
        position_1 = Position(1, 2, 3, time_point_number=1)
        position_2 = Position(4, 5, 6, time_point_number=2)
        experiment.positions.add(position_1)
        experiment.positions.add(position_2)
        experiment.links.add_link(position_1, position_2)

        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            io.save_data_to_json(experiment, file, write_new_format=True)

            with open(file) as handle:
                data = json.load(handle)
                self.assertEqual(expected_data["version"], data["version"])
                self.assertEqual(expected_data["positions"], data["positions"])
                self.assertEqual(expected_data["tracks"], data["tracks"])

    def test_save_format_with_meta(self):
        expected_data = {'version': 'v2',
                         'positions': [
                             {'time_point': 1, 'coords_xyz_px': [[1.0, 2.0, 3.0]], 'position_meta': {'test_key': [1]}},
                             {'time_point': 2, 'coords_xyz_px': [[4.0, 5.0, 6.0]], 'position_meta': {'test_key': [2]}},
                             {'time_point': 3, 'coords_xyz_px': [[7.0, 8.0, 9.0]], 'position_meta': {'other_key': [3]}}
                         ],
                         'tracks': [
                             {'time_point_start': 1,
                              'coords_xyz_px': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                              'link_meta': {'test_key': [None, 4]}}
                         ]}

        experiment = Experiment()
        position_1 = Position(1, 2, 3, time_point_number=1)
        position_2 = Position(4, 5, 6, time_point_number=2)
        position_3 = Position(7, 8, 9, time_point_number=3)
        experiment.positions.add(position_1)
        experiment.positions.add(position_2)
        experiment.positions.add(position_3)
        experiment.links.add_link(position_1, position_2)
        experiment.links.add_link(position_2, position_3)
        experiment.positions.set_position_data(position_1, "test_key", 1)
        experiment.positions.set_position_data(position_2, "test_key", 2)
        experiment.positions.set_position_data(position_3, "other_key", 3)
        experiment.link_data.set_link_data(position_2, position_3, "test_key", 4)
        # Between position_1 and position_2 there is no link data, so then test_key should become None there

        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            io.save_data_to_json(experiment, file, write_new_format=True)

            with open(file) as handle:
                data = json.load(handle)
                print(data)
                self.assertEqual(expected_data["version"], data["version"])
                self.assertEqual(expected_data["positions"], data["positions"])
                self.assertEqual(expected_data["tracks"], data["tracks"])

    def test_loading_new_format(self):
        input_data = {'version': 'v2',
                      'positions': [
                          {'time_point': 1, 'coords_xyz_px': [[1.0, 2.0, 3.0]], 'position_meta': {'test_key': [1]}},
                          {'time_point': 2, 'coords_xyz_px': [[4.0, 5.0, 6.0]], 'position_meta': {'test_key': [2]}},
                          {'time_point': 3, 'coords_xyz_px': [[7.0, 8.0, 9.0]], 'position_meta': {'other_key': [3]}}
                      ],
                      'tracks': [
                          {'time_point_start': 1,
                           'coords_xyz_px': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                           'link_meta': {'test_key': [None, 4]}}
                      ]}
        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            with open(file, 'w') as handle:
                json.dump(input_data, handle)

            experiment = io.load_data_file(file)
            self.assertEqual(3, len(experiment.positions))
            self.assertEqual(2, len(experiment.links))

            # Position meta
            self.assertEqual(1, experiment.positions.get_position_data(Position(1, 2, 3, time_point_number=1),
                                                                           "test_key"))
            self.assertEqual(3, experiment.positions.get_position_data(Position(7, 8, 9, time_point_number=3),
                                                                           "other_key"))

            # Link meta
            self.assertEqual(4, experiment.link_data.get_link_data(Position(4, 5, 6, time_point_number=2),
                                                                   Position(7, 8, 9, time_point_number=3), "test_key"))

    def test_loading_old_format(self):
        # To test that support for the old format is still there
        input_data = {'version': 'v1',
                      'positions': {'1': [[1.0, 2.0, 3.0]], '2': [[4.0, 5.0, 6.0]], '3': [[7.0, 8.0, 9.0]]},
                      'links': {
                          'directed': False,
                          'multigraph': False,
                          'graph': {},
                          'nodes': [{'id': {'x': 1.0, 'y': 2.0, 'z': 3.0, '_time_point_number': 1}, 'test_key': 1},
                                    {'id': {'x': 4.0, 'y': 5.0, 'z': 6.0, '_time_point_number': 2}, 'test_key': 2},
                                    {'id': {'x': 7.0, 'y': 8.0, 'z': 9.0, '_time_point_number': 3}, 'other_key': 3}],
                          'links': [
                                {'source': {'x': 1.0, 'y': 2.0, 'z': 3.0, '_time_point_number': 1},
                                 'target': {'x': 4.0, 'y': 5.0, 'z': 6.0, '_time_point_number': 2}},
                                {'source': {'x': 4.0, 'y': 5.0, 'z': 6.0, '_time_point_number': 2},
                                 'target': {'x': 7.0, 'y': 8.0, 'z': 9.0, '_time_point_number': 3}, 'test_key': 4}]}}

        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            with open(file, 'w') as handle:
                json.dump(input_data, handle)

            experiment = io.load_data_file(file)
            self.assertEqual(3, len(experiment.positions))
            self.assertEqual(2, len(experiment.links))

            # Position meta
            self.assertEqual(1, experiment.positions.get_position_data(Position(1, 2, 3, time_point_number=1),
                                                                           "test_key"))
            self.assertEqual(3, experiment.positions.get_position_data(Position(7, 8, 9, time_point_number=3),
                                                                           "other_key"))

            # Link meta
            self.assertEqual(4, experiment.link_data.get_link_data(Position(4, 5, 6, time_point_number=2),
                                                                   Position(7, 8, 9, time_point_number=3), "test_key"))

    def test_loading_with_min_time_point(self):
        experiment = Experiment()
        position_1 = Position(1, 2, 3, time_point_number=1)
        position_2 = Position(4, 5, 6, time_point_number=2)
        position_3 = Position(7, 8, 9, time_point_number=3)
        experiment.positions.add(position_1)
        experiment.positions.add(position_2)
        experiment.positions.add(position_3)
        experiment.links.add_link(position_1, position_2)
        experiment.links.add_link(position_2, position_3)
        experiment.link_data.set_link_data(position_1, position_2, "test_key", 1)
        experiment.link_data.set_link_data(position_2, position_3, "test_key", 2)

        with TemporaryDirectory() as directory:
            file = os.path.join(directory, "test." + io.FILE_EXTENSION)
            io.save_data_to_json(experiment, file, write_new_format=True)

            experiment = io.load_data_file(file, min_time_point=2, max_time_point=3)
            self.assertEqual(2, len(experiment.positions))
            self.assertEqual(1, len(experiment.links))
            self.assertEqual(2, experiment.link_data.get_link_data(position_2, position_3, "test_key"))
