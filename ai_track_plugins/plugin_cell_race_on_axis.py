"""Creates a movie showing the progress of individual cells over the crypt-villus axis."""
from typing import Dict, Any, List
from numpy import ndarray
import numpy
from tifffile import tifffile

from ai_track.core import UserError
from ai_track.core.links import Links, LinkingTrack
from ai_track.core.spline import SplineCollection
from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.linking_analysis.lineage_finder import LineageTree


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Gtaph//Histogram": lambda: _make_movement_image(window),
    }

def _make_movement_image(window: Window):
    experiment = window.get_experiment()
    output_file = dialog.prompt_save_file("Output file", [("TIFF file", "*.tif")])
    if output_file is None:
        return
    _draw_movement_image(experiment.links, experiment.splines, output_file=output_file)

def _draw_movement_image(links: Links, splines: SplineCollection, *, output_file: str, canvas_width: int = 450):
    if splines.first_time_point_number() is None:
        raise UserError("No data axes found", "No data axes were found. You need to draw the axes on which cells move.")
    time_point_count = splines.last_time_point_number() - splines.first_time_point_number() + 1

    links.sort_tracks_by_x()
    lineages = [LineageTree(track) for track in links.find_starting_tracks()
                if len(track.get_next_tracks()) > 0 or len(track) > 0.66 * time_point_count]
    canvas_height = sum((lineage.plotting_size + 10 for lineage in lineages))


    full_canvas = numpy.full((time_point_count, canvas_height, canvas_width, 3), fill_value=255, dtype=numpy.uint8)
    gray_canvas = numpy.copy(full_canvas[0])  # The gray lines background, gets fuller on each iteration

    for time_point in splines.time_points():
        print("Working on time point", time_point.time_point_number())
        time_point_number = time_point.time_point_number()

        canvas_index = time_point_number - splines.first_time_point_number()
        canvas = full_canvas[canvas_index]
        canvas[:] = gray_canvas  # Use the current gray canvas as a background

        y_offset = 0
        for lineage in lineages:
            color = lineage.get_color(links)
            tracks = lineage.get_tracks_at_time_point_number(time_point_number)
            lineage_width = lineage.plotting_size

            for index, track in enumerate(tracks):
                position = track.find_position_at_time_point_number(time_point_number)
                axis_position = splines.to_position_on_original_axis(links, position)
                if axis_position is None or axis_position.pos >= canvas_width:
                    if axis_position is not None:
                        print(axis_position.pos)
                    continue

                plotting_x = int(axis_position.pos)
                plotting_y = y_offset + index

                canvas[plotting_y - 2: plotting_y + 3, plotting_x - 2 : plotting_x + 3, 0] = int(color[0] * 255)
                canvas[plotting_y - 2: plotting_y + 3, plotting_x - 2 : plotting_x + 3, 1] = int(color[1] * 255)
                canvas[plotting_y - 2: plotting_y + 3, plotting_x - 2 : plotting_x + 3, 2] = int(color[2] * 255)
                gray_canvas[plotting_y, plotting_x, 0] = 200
                gray_canvas[plotting_y, plotting_x, 1] = 200
                gray_canvas[plotting_y, plotting_x, 2] = 200

            y_offset += lineage_width + 10

    tifffile.imsave(output_file, full_canvas)
