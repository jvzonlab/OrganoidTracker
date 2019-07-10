from typing import Union, Dict, List, Optional

import numpy
from numpy import ndarray

from ai_track.core import TimePoint
from ai_track.core.position_collection import PositionCollection
from ai_track.core.position import Position

_MAX_SHOWN = 15


class Category:
    _name: str

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name  # Immutable

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if other is self:
            return True
        if isinstance(other, Category):
            return other.name == self.name
        return False

    def __repr__(self):
        return f"Category({repr(self._name)})"


class Statistics:
    """Used to plot the true positives, false positives, false negatives, recall, precisiion and F1"""

    # None of these variables should be changed, but I'm too lazy to make them all read-only
    time_point_numbers: ndarray  # Int array, used on x axis
    true_positives: ndarray
    false_positives: ndarray
    false_negatives: ndarray
    precision: ndarray
    recall: ndarray
    f1_score: ndarray

    precision_overall: float
    recall_overall: float
    f1_score_overall: float

    def __init__(self, first_time_point_number: int, true_positives: ndarray, false_positives: ndarray, false_negatives: ndarray):
        self.time_point_numbers = numpy.arange(first_time_point_number, first_time_point_number + len(true_positives))
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives

        self.precision = true_positives / (true_positives + false_positives)
        self.recall = true_positives / (true_positives + false_negatives)
        self.f1_score =(2 * self.precision * self.recall) / (self.precision + self.recall)

        true_positives_overall = true_positives.sum()
        false_positives_overall = false_positives.sum()
        false_negatives_overall = false_negatives.sum()
        self.precision_overall = true_positives_overall / (false_negatives_overall + false_positives_overall)
        self.recall_overall = true_positives_overall / (true_positives_overall + false_negatives_overall)
        self.f1_score_overall = (2 * self.precision_overall * self.recall_overall) / \
                                (self.precision_overall + self.recall_overall)

    def debug_plot(self):
        """Shows a Matplotlib plot. Must only be called from the command line."""
        import matplotlib.pyplot as plt
        plt.plot(self.time_point_numbers, self.recall, label="Recall")
        plt.plot(self.time_point_numbers, self.precision, label="Precision")
        plt.plot(self.time_point_numbers, self.f1_score, label="F1 score")
        plt.xlabel("Time point")
        plt.title(f"Recall: {self.recall.mean():.02f}, Precision: {self.precision.mean():.02f},"
                  f" F1 score: {self.f1_score.mean():.02f}")
        plt.legend()
        plt.show()


class ComparisonReport:

    title: str = "Comparison"
    summary: str = ""
    _details_by_position: Dict[Position, str]
    _positions_by_category: Dict[Category, PositionCollection]

    def __init__(self):
        self._details_by_position = dict()
        self._positions_by_category = dict()

    def add_data(self, category: Category, position: Position, *details: Union[str, Position]):
        """Adds a data point. The """
        if category not in self._positions_by_category:
            self._positions_by_category[category] = PositionCollection()

        if details:
            self._details_by_position[position] = " ".join(str(detail) for detail in details)
        self._positions_by_category[category].add(position)

    def __str__(self):
        report = self.title + "\n"
        report += ("-" * len(self.title)) + "\n\n"
        report += self.summary + "\n"
        for category, positions in self._positions_by_category.items():
            count = len(positions)
            report += "\n" + category.name + ": ("+str(count)+")\n"

            i = 0
            for position in positions:
                position_str = str(position)
                details = self._details_by_position.get(position)
                if details is not None:
                    position_str += " - " + details
                report += "* " + position_str + "\n"
                i += 1
                if i > _MAX_SHOWN:
                    report += "... " + str(count - _MAX_SHOWN) + " entries not shown\n"
                    break
        return report

    def calculate_statistics(self, true_positives_cat: Category, false_positives_cat: Category,
                             false_negatives_cat: Category) -> Statistics:
        """Calculate statistics using the given categories as false/true positives/negatives."""
        min_time_point_number = min(self._positions_by_category[true_positives_cat].first_time_point_number(),
                                    self._positions_by_category[false_positives_cat].first_time_point_number(),
                                    self._positions_by_category[false_negatives_cat].first_time_point_number())
        max_time_point_number = max(self._positions_by_category[true_positives_cat].last_time_point_number(),
                                    self._positions_by_category[false_positives_cat].last_time_point_number(),
                                    self._positions_by_category[false_negatives_cat].last_time_point_number())

        time_point_count = max_time_point_number - min_time_point_number + 1
        true_positives = numpy.ones(time_point_count, dtype=numpy.uint16)
        false_positives = numpy.ones(time_point_count, dtype=numpy.uint16)
        false_negatives = numpy.ones(time_point_count, dtype=numpy.uint16)
        for i in range(time_point_count):
            time_point = TimePoint(i + min_time_point_number)
            true_positives[i] = len(self._positions_by_category[true_positives_cat].of_time_point(time_point))
            false_positives[i] = len(self._positions_by_category[false_positives_cat].of_time_point(time_point))
            false_negatives[i] = len(self._positions_by_category[false_negatives_cat].of_time_point(time_point))
        return Statistics(min_time_point_number, true_positives, false_positives, false_negatives)

    def count(self, category: Category) -> int:
        """Gets how many positions are stored in the given category."""
        positions = self._positions_by_category.get(category)
        if positions is None:
            return 0
        return len(positions)
