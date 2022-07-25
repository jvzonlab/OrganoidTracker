"""
With this package, you can compare two experiments. Useful to check whether your cell tracking algorithm works.

>>> from organoid_tracker.core.experiment import Experiment
>>> ground_truth = Experiment()  # Placeholder
>>> tracking_output = Experiment()  # Placeholder
>>>
>>> from organoid_tracker.comparison import links_comparison
>>> report = links_comparison.compare_links(ground_truth, tracking_output)
>>>
>>> # Save to a file that OrganoidTracker can load in the Tools menu
>>> from organoid_tracker.comparison import report_json_io
>>> output_file = "my_output_file.json"
>>> report_json_io.save_report(report, output_file)
>>>
>>> # Show graphs
>>> report.calculate_time_correctness_statistics().debug_plot()
>>> report.calculate_z_correctness_statistics().debug_plot()
"""
