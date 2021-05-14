from typing import Dict, Optional
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution


def load_data_file(file_name: str, min_time_point: int, max_time_point: int, experiment: Optional[Experiment] = None
                   ) -> Experiment:
    """Loads an XML file in the TrackMate format."""
    if experiment is None:
        experiment = Experiment()

    tree = ElementTree.parse(file_name)
    root = tree.getroot()
    settings = root.find("Settings")
    if settings is not None:
        _read_resolution(experiment, settings)
    model = root.find("Model")

    # Validate
    if model.attrib["spatialunits"] not in {"µm", "pixel"}:
        raise ValueError("Can only handle coordinates in micrometers and pixels. Cannot handle \""
                         + str(model.attrib["spatialunits"]) + "\".")

    # Read all spots
    spot_dictionary = _read_spots(experiment, model, min_time_point, max_time_point)
    all_tracks = model.find("AllTracks")
    links = experiment.links
    for track in all_tracks.findall("Track"):
        for edge in track.findall("Edge"):
            source = spot_dictionary.get(int(edge.attrib["SPOT_SOURCE_ID"]))
            target = spot_dictionary.get(int(edge.attrib["SPOT_TARGET_ID"]))
            if source is None or target is None:
                continue  # Outside of time range

            if source.time_point_number() > target.time_point_number():
                source, target = target, source  # Make sure source comes first in time

            while source.time_point_number() < target.time_point_number() - 1:
                # Add extra positions in case a time point is skipped
                temp_position = source.with_time_point_number(source.time_point_number() + 1)
                links.add_link(source, temp_position)
                source = temp_position
            links.add_link(source, target)
    return experiment


def _read_spots(experiment: Experiment, model: Element, min_time_point: int, max_time_point: int) -> Dict[int, Position]:
    """Reads all spots in the XML tree, and adds them to the experiment. Returns a Dict with a
    spot id -> Position mapping, which will be used for linking."""
    spot_dictionary = dict()
    positions = experiment.positions

    # TrackMate stores positions in the resolution, we store it in pixels. Correct for that.
    x_res, y_res, z_res = 1, 1, 1
    try:
        resolution = experiment.images.resolution()
        x_res = resolution.pixel_size_x_um
        y_res = resolution.pixel_size_y_um
        z_res = resolution.pixel_size_z_um
    except UserError:
        pass  # Ignore, seems TrackMate is actually storing positions.
    coords_in_px = model.attrib["spatialunits"] == "pixel"

    all_spots = model.find("AllSpots")
    for frame in all_spots.findall("SpotsInFrame"):
        time_point_number = int(frame.attrib["frame"])
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue  # Skip this time point
        time_point = TimePoint(time_point_number)
        for spot in frame.findall("Spot"):
            x = float(spot.attrib["POSITION_X"])
            y = float(spot.attrib["POSITION_Y"])
            z = float(spot.attrib["POSITION_Z"])
            if not coords_in_px:
                x /= x_res
                y /= y_res
                z /= z_res

            id = int(spot.attrib["ID"])
            position = Position(x, y, z, time_point=time_point)
            positions.add(position)
            spot_dictionary[id] = position
    return spot_dictionary


def _read_resolution(experiment: Experiment, settings: Element):
    image_data = settings.find("ImageData")
    if image_data is not None:
        x_res = float(image_data.attrib["pixelwidth"])
        y_res = float(image_data.attrib["pixelheight"])
        z_res = float(image_data.attrib["voxeldepth"])
        t_res = float(image_data.attrib["timeinterval"]) if "timeinterval" in image_data.attrib else 0
        experiment.images.set_resolution(ImageResolution(x_res, y_res, z_res, t_res))
