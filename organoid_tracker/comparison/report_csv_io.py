from organoid_tracker.comparison.report import ComparisonReport


def save_report_time_statistics(report: ComparisonReport, file_name: str):
    """Saves the number of positions in each category for each time point to a CSV file."""
    categories = report.get_categories()

    with open(file_name, "w") as handle:
        # Write headers
        handle.write("Time point")
        for category in categories:
            handle.write("," + category.name)

        # Write rows
        for time_point in report.time_points():
            handle.write(f"\n{time_point.time_point_number()}")
            for category in categories:
                handle.write(f",{report.count_positions(category, time_point=time_point)}")

        handle.write("\n")
