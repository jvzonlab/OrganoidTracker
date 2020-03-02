import json
from typing import Dict, Any, List, Union

from organoid_tracker.comparison.report import ComparisonReport, Category, Details
from organoid_tracker.core.position import Position


def save_report(report: ComparisonReport, file: str):
    """Saves the report to the given file. Make sure the parent directory of the file already exists."""
    output = {
        "title": report.title,
        "summary": report.summary,
        "parameters": dict(report.recorded_parameters()),
        "categories": [_category_to_json(report, category) for category in report.get_categories()]
    }
    with open(file, "w", encoding="utf8") as file_handle:
        json.dump(output, file_handle)


def _category_to_json(report: ComparisonReport, category: Category) -> Dict[str, Any]:
    """Covnerts all entries in a report category to a JSON structure."""
    return {
        "name": category.name,
        "entries": [_entry_to_json(position, details) for position, details in report.get_entries(category)]
    }


def _entry_to_json(position: Position, details: Details) -> List[Union[str, List[float]]]:
    """Converts a single entry (position, details) to a JSON list of lists (=positions) and strings."""
    return_list = [[position.x, position.y, position.z, position.time_point_number()]]
    for detail in details.details:
        if isinstance(detail, Position):
            return_list.append([detail.x, detail.y, detail.z, detail.time_point_number()])
        else:
            return_list.append(str(detail))
    return return_list


def load_report(file: str) -> ComparisonReport:
    """Reconstructs a report from a JSON file."""
    with open(file, "r", encoding="utf8") as file_handle:
        input = json.load(file_handle)

    parameters = input["parameters"] if "parameters" in input else dict()
    report = ComparisonReport(**parameters)

    report.title = input["title"]
    report.summary = input["summary"]
    for category_input in input["categories"]:
        # Read each category
        category = Category(category_input["name"])
        for entry_input in category_input["entries"]:
            # Read each data entry in every category
            position_input = entry_input[0]
            position = Position(position_input[0], position_input[1], position_input[2], time_point_number=position_input[3])
            details = list()
            for detail_input in entry_input[1:]:
                # Read the details of every data entry
                if isinstance(detail_input, str):
                    details.append(detail_input)
                else:  # not a str, so must be a list
                    details.append(Position(detail_input[0], detail_input[1], detail_input[2], time_point_number=detail_input[3]))
            report.add_data(category, position, *details)
    return report
