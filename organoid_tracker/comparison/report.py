from typing import Union, Dict, List, Optional, Tuple, Iterable

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import DataType

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
    x_axis_numbers: ndarray  # Int array, used on x axis
    x_label: str
    true_positives: ndarray
    false_positives: ndarray
    false_negatives: ndarray
    precision: ndarray
    recall: ndarray
    f1_score: ndarray

    precision_overall: float
    recall_overall: float
    f1_score_overall: float

    def __init__(self, first_x_axis_number: int, x_label: str, true_positives: ndarray, false_positives: ndarray,
                 false_negatives: ndarray):
        self.x_axis_numbers = numpy.arange(first_x_axis_number, first_x_axis_number + len(true_positives))
        self.x_label = x_label
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives

        self.precision = true_positives / (true_positives + false_positives)
        self.recall = true_positives / (true_positives + false_negatives)
        self.f1_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

        true_positives_overall = true_positives.sum()
        false_positives_overall = false_positives.sum()
        false_negatives_overall = false_negatives.sum()
        self.precision_overall = true_positives_overall / (true_positives_overall + false_positives_overall)
        self.recall_overall = true_positives_overall / (true_positives_overall + false_negatives_overall)
        self.f1_score_overall = (2 * self.precision_overall * self.recall_overall) / \
                                (self.precision_overall + self.recall_overall)

    def debug_plot(self):
        """Shows a Matplotlib plot. Must only be called from the command line."""
        import matplotlib.pyplot as plt
        value_count = self.recall.size - numpy.isnan(self.recall).sum()
        symbol = "o" if value_count < 25 else "-"
        plt.plot(self.x_axis_numbers, self.recall, symbol, label="Recall")
        plt.plot(self.x_axis_numbers, self.precision, symbol, label="Precision")
        plt.plot(self.x_axis_numbers, self.f1_score, symbol, label="F1 score")
        plt.xlabel(self.x_label)
        plt.title(f"Recall: {self.recall_overall:.02f}, Precision: {self.precision_overall:.02f},"
                  f" F1 score: {self.f1_score_overall:.02f}")
        plt.tick_params(axis="both", direction="in")
        plt.legend()
        plt.show()


class Details:
    """Represents the details of a certain entry in the report."""
    _details: Tuple[Union[str, Position], ...]

    def __init__(self, *detail: Union[str, Position]):
        """Creates a new detail entry. Positions will be auto-linked if viewed in the visualizer."""
        self._details = tuple(detail)

    @property
    def details(self) -> Tuple[Union[str, Position]]:
        """The details, cannot be modified."""
        return self._details

    def __str__(self) -> str:
        return " ".join(str(detail) for detail in self._details)

    def __repr__(self) -> str:
        return "Details(" + " ".join(repr(detail) for detail in self._details) + ")"


class ComparisonReport:
    title: str = "Comparison"
    summary: str = ""
    _recorded_parameters: Dict[str, DataType]
    _details_by_category_and_position: Dict[Category, Dict[Position, Details]]
    _positions_by_category: Dict[Category, PositionCollection]

    def __init__(self, **parameters: DataType):
        self._recorded_parameters = parameters
        self._details_by_category_and_position = dict()
        self._positions_by_category = dict()

    def add_data(self, category: Category, position: Position, *details: Union[str, Position]):
        """Adds a data point. The """
        if category not in self._positions_by_category:
            self._positions_by_category[category] = PositionCollection()
        self._positions_by_category[category].add(position)

        if details:
            if category not in self._details_by_category_and_position:
                self._details_by_category_and_position[category] = dict()
            self._details_by_category_and_position[category][position] = Details(*details)

    def delete_data(self, category: Category, position: Position):
        """Deletes all data of the given position."""
        positions_in_category = self._positions_by_category.get(category)
        if positions_in_category is None:
            return
        positions_in_category.detach_position(position)

        # Also delete the details
        details_in_category = self._details_by_category_and_position.get(category)
        if details_in_category is not None and position in details_in_category:
            del details_in_category[position]

    def get_categories(self) -> Iterable[Category]:
        """Gets all categories that are used in this report."""
        return tuple(self._positions_by_category.keys())

    def get_category_by_name(self, name: str) -> Optional[Category]:
        """Gets the category with the given name."""
        for category in self._positions_by_category.keys():
            if category.name == name:
                return category
        return None

    def __str__(self):
        report = self.title + "\n"
        report += ("=" * len(self.title)) + "\n\n"
        if len(self.summary) > 0:
            report += self.summary + "\n"
        if not len(self._recorded_parameters) == 0:
            report += "\n"
            for parameter_name, parameter_value in self._recorded_parameters.items():
                report += f"    {parameter_name} = {parameter_value}\n"

        for category, positions in self._positions_by_category.items():
            details_by_position = self._details_by_category_and_position.get(category)
            if details_by_position is None:
                details_by_position = dict()  # Supply an empty map to prevent errors

            count = len(positions)
            header = category.name + ": (" + str(count) + ")"
            report += "\n" + header + "\n" + ("-" * len(header)) + "\n"

            i = 0
            for position in positions:
                position_str = str(position)
                details = details_by_position.get(position)
                if details is not None:
                    position_str += " - " + str(details)
                report += "* " + position_str + "\n"
                i += 1
                if i > _MAX_SHOWN:
                    report += "... " + str(count - _MAX_SHOWN) + " entries not shown\n"
                    break
        return report

    def calculate_time_statistics(self, true_positives_cat: Category, false_positives_cat: Category,
                                  false_negatives_cat: Category) -> Statistics:
        """Calculate statistics using the given categories as false/true positives/negatives."""
        empty = PositionCollection()  # Used for non-existing categories

        min_time_point_number = min_none(self._positions_by_category.get(true_positives_cat, empty).first_time_point_number(),
                                         self._positions_by_category.get(false_positives_cat, empty).first_time_point_number(),
                                         self._positions_by_category.get(false_negatives_cat, empty).first_time_point_number())
        max_time_point_number = max_none(self._positions_by_category.get(true_positives_cat, empty).last_time_point_number(),
                                         self._positions_by_category.get(false_positives_cat, empty).last_time_point_number(),
                                         self._positions_by_category.get(false_negatives_cat, empty).last_time_point_number())

        time_point_count = max_time_point_number - min_time_point_number + 1
        true_positives = numpy.ones(time_point_count, dtype=numpy.uint16)
        false_positives = numpy.ones(time_point_count, dtype=numpy.uint16)
        false_negatives = numpy.ones(time_point_count, dtype=numpy.uint16)
        for i in range(time_point_count):
            time_point = TimePoint(i + min_time_point_number)
            true_positives[i] = len(self._positions_by_category.get(true_positives_cat, empty).of_time_point(time_point))
            false_positives[i] = len(self._positions_by_category.get(false_positives_cat, empty).of_time_point(time_point))
            false_negatives[i] = len(self._positions_by_category.get(false_negatives_cat, empty).of_time_point(time_point))
        return Statistics(min_time_point_number, "Time point", true_positives, false_positives, false_negatives)

    def calculate_z_statistics(self, true_positives_cat: Category, false_positives_cat: Category,
                               false_negatives_cat: Category) -> Statistics:
        """Calculate statistics using the given categories as false/true positives/negatives."""
        min_z = min(self._positions_by_category[true_positives_cat].lowest_z(),
                    self._positions_by_category[false_positives_cat].lowest_z(),
                    self._positions_by_category[false_negatives_cat].lowest_z())
        max_time_point_number = max(self._positions_by_category[true_positives_cat].highest_z(),
                                    self._positions_by_category[false_positives_cat].highest_z(),
                                    self._positions_by_category[false_negatives_cat].highest_z())

        z_count = max_time_point_number - min_z + 1
        true_positives = numpy.ones(z_count, dtype=numpy.uint16)
        false_positives = numpy.ones(z_count, dtype=numpy.uint16)
        false_negatives = numpy.ones(z_count, dtype=numpy.uint16)
        for i in range(z_count):
            z = i + min_z
            true_positives[i] = sum(1 for _ in self._positions_by_category[true_positives_cat].nearby_z(z))
            false_positives[i] = sum(1 for _ in self._positions_by_category[false_positives_cat].nearby_z(z))
            false_negatives[i] = sum(1 for _ in self._positions_by_category[false_negatives_cat].nearby_z(z))
        return Statistics(min_z, "Z layer", true_positives, false_positives, false_negatives)

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point number at which data exits for at least one category. Returns None if therea is no
        data at all in this object."""
        first_time_point_number = None
        for category, items in self._positions_by_category.items():
            first_time_point_number = min_none(first_time_point_number, items.first_time_point_number())
        return first_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the first time point number at which data exits for at least one category. Returns None if therea is no
        data at all in this object."""
        last_time_point_number = None
        for category, items in self._positions_by_category.items():
            last_time_point_number = max_none(last_time_point_number, items.last_time_point_number())
        return last_time_point_number

    def time_points(self) -> Iterable[TimePoint]:
        """Returns all time points from the first to the last, inclusive. Time points in between might not have data
        attached to them."""
        min_time_point_number = self.first_time_point_number()
        max_time_point_number = self.last_time_point_number()
        if min_time_point_number is None or max_time_point_number is None:
            return
        for time_point_number in range(min_time_point_number, max_time_point_number + 1):
            yield TimePoint(time_point_number)

    def get_entries(self, category: Category) -> Iterable[Tuple[Position, Details]]:
        """Gets all entries for the given category."""
        if category not in self._positions_by_category:
            return
        empty_details = Details()  # Reused to save memory

        details_by_position = self._details_by_category_and_position.get(category)
        if details_by_position is None:
            details_by_position = dict()  # Supply an empty map to prevent errors

        for position in self._positions_by_category[category]:
            details = details_by_position.get(position)
            if details is None:
                details = empty_details
            yield (position, details)

    def count_positions(self, category: Category, *, time_point: Optional[TimePoint] = None) -> int:
        """Gets how many entries there are in the given category, optionally at the given time point. Returns 0 if the
        given category is not used."""
        entries = self._positions_by_category.get(category)
        if entries is None:
            return 0
        return entries.count_positions(time_point=time_point)

    def get_positions(self, category: Category, *, time_point: Optional[TimePoint]) -> Iterable[Position]:
        """Gets all positions in the given category, optionally filtered to the given time point."""
        entries = self._positions_by_category.get(category)
        if entries is None:
            return []
        if time_point is not None:
            return entries.of_time_point(time_point)
        return entries

    def recorded_parameters(self) -> Iterable[Tuple[str, DataType]]:
        """Gets all parameters set for this comparison. Useful for reproducing this comparison."""
        yield from self._recorded_parameters.items()
