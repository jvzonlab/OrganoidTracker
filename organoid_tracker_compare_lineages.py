#!/usr/bin/env python3

"""Compares two linking results on the level of lineages, so that missing cell deaths and divisions will be reported.
The baseline links are assumed to be 100% correct, any deviations from that are counted as errors."""
from organoid_tracker.comparison import lineage_comparison, report_json_io
from organoid_tracker.config import ConfigFile, config_type_json_file
from organoid_tracker.imaging import io

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("compare_lineages")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

_automatic_links_file = config.get_or_prompt("automatic_links_file", "In what file are the new links stored?")
_baseline_links_file = config.get_or_prompt("baseline_links_file", "In what file are the original links stored?")
_max_distance_um = float(config.get_or_default("max_distance_um", str(5)))
_output_file = config.get_or_default("output_file", "", type=config_type_json_file)
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")
scratch_experiment = io.load_data_file(_automatic_links_file, _min_time_point, _max_time_point)
baseline_experiment = io.load_data_file(_baseline_links_file, _min_time_point, _max_time_point)

print("Comparing...")
report = lineage_comparison.compare_links(baseline_experiment, scratch_experiment, _max_distance_um)
if _output_file:
    report_json_io.save_report(report, _output_file)
print(report)

print("Done!")
