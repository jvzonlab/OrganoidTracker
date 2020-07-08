from typing import Iterable

from organoid_tracker.comparison import report_json_io
from organoid_tracker.comparison.report import ComparisonReport, Category
from organoid_tracker.config import ConfigFile, config_type_json_file
from organoid_tracker.imaging import io
from organoid_tracker.linking_analysis import cell_error_finder, linking_markers

TRUE_ERRORS = Category("True errors")
FALSE_ERRORS = Category("False errors")

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("validate_errors")
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
report = ComparisonReport()
for errored_position in linking_markers.find_errored_positions(automatic_experiment.position_data):
    if not corrected_experiment.positions.contains_position(errored_position):
        # Entire position was removed, so clearly the error was correct
        report.add_data(TRUE_ERRORS, errored_position, "was removed from/moved in corrected data")
        continue

    corrected_links_of_position = corrected_experiment.links.find_links_of(errored_position)
    original_links_of_position = automatic_experiment.links.find_links_of(errored_position)
    if corrected_links_of_position != original_links_of_position:
        report.add_data(TRUE_ERRORS, errored_position, "got its links changed")
    else:
        report.add_data(FALSE_ERRORS, errored_position, "had nothing changed in position or links")


def _iterable_len(iterable: Iterable) -> int:
    count = 0
    for _ in iterable:
        count += 1
    return count


print("True warnings:", _iterable_len(report.get_entries(TRUE_ERRORS)))
print("Superfluous warnings:", _iterable_len(report.get_entries(FALSE_ERRORS)))

if _output_file:
    print("Saved to", _output_file)
    report_json_io.save_report(report, _output_file)
