import csv
from typing import Dict, Any, Union, Optional

from organoid_tracker.core import UserError
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Import-Import positions//CSV, with pixel coordinates": lambda: _import_positions(window),
        "File//Import-Import positions//CSV, metadata for existing positions": lambda: _import_metadata(window)
    }


def _parse(value: str) -> Union[str, float, int]:
    if "." in value:
        try:
            return float(value)
        except ValueError:
            return value
    else:
        try:
            return int(value)
        except ValueError:
            return value


def _import_positions(window: Window):
    experiment = window.get_experiment()

    if not dialog.prompt_confirmation("Importing CSV metadata", "This function requires a single CSV file with"
                                                                " x, y, z (pixels) and t (time point) columns. Any other"
                                                                " columns can be used as metadata."):
        return
    if experiment.positions.has_positions():
        if not dialog.prompt_confirmation("Existing positions found",
                                          "Existing positions were found. Are you sure you want"
                                          " to import more positions? The two sets of positions"
                                          " will be merged."):
            return

    new_positions_count = _import_csv(window, new_positions=True)
    dialog.popup_message("Imported positions", f"Imported all positions. {new_positions_count} new positions"
                                               f" were added.")


def _import_metadata(window: Window):
    experiment = window.get_experiment()

    if not experiment.positions.has_positions():
        raise UserError("No existing positions", "This function adds metadata from a CSV file to existing positions."
                                                 " However, no existing positions were found.")
    if not dialog.prompt_confirmation("Importing CSV metadata", "This function requires a single CSV file with"
                                                                " precise values in x, y, z and t columns. We expect"
                                                                " pixel coordinates and time points (so no micrometers"
                                                                " or minutes). The other columns can then be used as"
                                                                " metadata."):
        return

    skipped_positions_count = _import_csv(window, new_positions=False)
    if skipped_positions_count is None:
        return  # Cancelled
    if skipped_positions_count > 0:
        dialog.popup_message("Imported metadata",
                             f"Imported all metadata. {skipped_positions_count} positions did not exist,"
                             f" so their metadata could not be imported.")
    else:
        dialog.popup_message("Imported metadata",
                             f"Imported all metadata. If you double-click a position, the metadata should now show up.")


def _import_csv(window: Window, *, new_positions: bool = True) -> Optional[int]:
    experiment = window.get_experiment()

    csv_file = dialog.prompt_load_file("CSV file", [("CSV file", "*.csv")])
    if csv_file is None:
        return None

    with open(csv_file) as handle:
        reader = csv.reader(handle)
        headers = next(reader)

        useful_headers = headers.copy()
        if '' in useful_headers:
            useful_headers.remove('')
        if 't' not in useful_headers or 'x' not in useful_headers or 'y' not in useful_headers or 'z' not in useful_headers:
            raise UserError("Missing columns", "The CSV file should contain columns named 'x', 'y', 'z' and 't',"
                                               " but contains only '" + "', '".join(headers) + "'.")
        useful_headers.remove('x')
        useful_headers.remove('y')
        useful_headers.remove('z')
        useful_headers.remove('t')

        selected_option_indices = option_choose_dialog.prompt_list_multiple("Metadata columns",
                                                                            "We found columns available for import as"
                                                                            " metadata. Do you want to import those?",
                                                                            label="Metadata to import:",
                                                                            options=useful_headers)
        if selected_option_indices is None:
            selected_option_indices = []

        # Switch to option names and indices of original headers array
        selected_options = [useful_headers[index] for index in selected_option_indices]
        selected_option_indices = [headers.index(option) for option in selected_options]
        x_index = headers.index('x')
        y_index = headers.index('y')
        z_index = headers.index('z')
        t_index = headers.index('t')

        positions_that_did_not_exist = 0
        for row in reader:
            x = float(row[x_index])
            y = float(row[y_index])
            z = float(row[z_index])
            t = int(row[t_index])
            position = Position(x, y, z, time_point_number=t)

            if not experiment.positions.contains_position(position):
                positions_that_did_not_exist += 1
                if new_positions:
                    experiment.positions.add(position)

            for metadata_id, metadata_name in zip(selected_option_indices, selected_options):
                experiment.position_data.set_position_data(position, metadata_name, _parse(row[metadata_id]))
    window.redraw_data()

    return positions_that_did_not_exist
