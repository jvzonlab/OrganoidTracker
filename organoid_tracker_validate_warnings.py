from typing import Iterable

from organoid_tracker.comparison import report_json_io
from organoid_tracker.comparison.report import ComparisonReport, Category
from organoid_tracker.config import ConfigFile, config_type_json_file
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io
from organoid_tracker.linking_analysis import linking_markers

TRUE_ERRORS = Category("True errors")
FALSE_ERRORS = Category("False errors")

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("validate_warnings")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

_automatic_links_file = config.get_or_prompt("automatic_links_file", "In what file are the automatic tracks stored?")
_corrected_links_file = config.get_or_prompt("corrected_links_file", "In what file are the corrected tracks stored?")
_output_file = config.get_or_default("output_file", "", type=config_type_json_file)
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")
automatic_experiment = io.load_data_file(_automatic_links_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
corrected_experiment = io.load_data_file(_corrected_links_file, min_time_point=_min_time_point, max_time_point=_max_time_point)

print("Comparing...")
def _translate(position_a: Position, experiment_a: Experiment, experiment_b: Experiment) -> Position:
    """Corrects the position for two experiments that have different offests."""
    time_point = position_a.time_point()
    position = position_a - experiment_a.images.offsets.of_time_point(time_point)
    position_b = position + experiment_b.images.offsets.of_time_point(time_point)
    return position_b


def _iterable_len(iterable: Iterable) -> int:
    """Like len(..), but for iterables."""
    count = 0
    for _ in iterable:
        count += 1
    return count


report = ComparisonReport()
for errored_position in linking_markers.find_errored_positions(automatic_experiment.position_data):
    translated_position = _translate(errored_position, automatic_experiment, corrected_experiment)

    if not corrected_experiment.positions.contains_position(translated_position):
        # Entire position was removed, so clearly the error was correct
        report.add_data(TRUE_ERRORS, errored_position, "was removed from/moved in corrected data")
        continue

    corrected_links_of_position = {_translate(pos, corrected_experiment, automatic_experiment) for pos in
                                   corrected_experiment.links.find_links_of(translated_position)}
    original_links_of_position = automatic_experiment.links.find_links_of(errored_position)
    if corrected_links_of_position != original_links_of_position:
        report.add_data(TRUE_ERRORS, errored_position, "got its links changed")
    else:
        report.add_data(FALSE_ERRORS, errored_position, "had nothing changed in position or links")


print("True warnings:", _iterable_len(report.get_entries(TRUE_ERRORS)))
print("Superfluous warnings:", _iterable_len(report.get_entries(FALSE_ERRORS)))
report.calculate_time_statistics(TRUE_ERRORS, FALSE_ERRORS, Category("False negatives are not identified")).debug_plot()

if _output_file:
    print("Saved to", _output_file)
    report_json_io.save_report(report, _output_file)

print("Done!")