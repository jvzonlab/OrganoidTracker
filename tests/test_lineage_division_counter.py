from unittest import TestCase

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.linking_analysis import lineage_division_counter


class TestLineageDivisionCounter(TestCase):

    def test_get_number_of_cells_at_end(self):
        # Build a small lineage tree with a mother cell and two daughter cells.
        pos_mother_1 = Position(0, 0, 0, time_point_number=0)
        pos_mother_2 = Position(0, 0, 0, time_point_number=1)
        pos_mother_3 = Position(0, 0, 0, time_point_number=2)
        pos_daughter_a_1 = Position(1, 0, 0, time_point_number=3)
        pos_daughter_a_2 = Position(1, 0, 0, time_point_number=4)
        pos_daughter_b_1 = Position(2, 0, 0, time_point_number=3)
        pos_daughter_b_2 = Position(2, 0, 0, time_point_number=4)


        links = Links()
        links.add_link(pos_mother_1, pos_mother_2)
        links.add_link(pos_mother_2, pos_mother_3)

        links.add_link(pos_mother_3, pos_daughter_a_1)
        links.add_link(pos_daughter_a_1, pos_daughter_a_2)

        links.add_link(pos_mother_3, pos_daughter_b_1)
        links.add_link(pos_daughter_b_1, pos_daughter_b_2)

        mother_track = links.get_track(pos_mother_1)

        last_time_point_number_of_experiment = 4
        self.assertEqual(2 , lineage_division_counter.get_number_of_cells_at_end(mother_track, last_time_point_number_of_experiment))



