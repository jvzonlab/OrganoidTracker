"""
The TrackMate file format works as follows:

- XML file with a root tag "TrackMate"  <TrackMate version="3.6.0">
  - Subtag "Model": <Model spatialunits="µm" timeunits="frame">
    - Some default tags:
        <FeatureDeclarations>
          <SpotFeatures />
          <EdgeFeatures />
          <TrackFeatures />
        </FeatureDeclarations>
    - The positions:
      <AllSpots nspots="1234">
        <SpotsInFrame frame="1">
          <Spot FRAME="1" ID="1668" POSITION_T="1.0" POSITION_X="49.24473640416" POSITION_Y="51.907814350324756"
           POSITION_Z="64.18574949599984" QUALITY="-1.0" RADIUS="3" VISIBILITY="1" name="1668" />
          etc.
        </SpotsInFrame>
      </AllSpots>
    - The links:
      <AllTracks>
        <Track TRACK_ID="1" name="1">
          <Edge SPOT_SOURCE_ID="1668" SPOT_TARGET_ID="1693" />
          ...
        </Track>
        ..
      </AllTracks>
      <FilteredTracks>
        <TrackID TRACK_ID="1" />
      </FilteredTracks>
  - Subtag "Settings"
    - Subtag <ImageData filename="a.xml" folder="" height="512" nframes="326" nslices="32" pixelheight="0.32"
              pixelwidth="0.32" voxeldepth="2.0" timeinterval="12" width="512" />
    - Other tags to keep Trackmate happy:
        <InitialSpotFilter feature="QUALITY" isabove="true" value="0.0" />
        <SpotFilterCollection />
        <TrackFilterCollection />
        <AnalyzerCollection>
          <SpotAnalyzers />
          <EdgeAnalyzers>
            <Analyzer key="Edge target" />
          </EdgeAnalyzers>
          <TrackAnalyzers />
        </AnalyzerCollection>
    -
"""

from typing import Dict, Optional
import xml.etree.ElementTree as ElementTreeLib
from xml.etree.ElementTree import Element, SubElement, Comment

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution


def load_data_file(file_name: str, min_time_point: int, max_time_point: int, experiment: Optional[Experiment] = None
                   ) -> Experiment:
    """Loads an XML file in the TrackMate format."""
    if experiment is None:
        experiment = Experiment()

    tree = ElementTreeLib.parse(file_name)
    root = tree.getroot()
    settings = root.find("Settings")
    if settings is not None:
        _read_resolution(experiment, settings)
    model = root.find("Model")

    # Validate
    if model.attrib["spatialunits"] not in {"µm", "pixel"}:
        raise ValueError("Can only handle coordinates in micrometers and pixels. Cannot handle \""
                         + str(model.attrib["spatialunits"]) + "\".")

    # Read all tracks
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


def _read_spots(experiment: Experiment, model: Element, min_time_point: int, max_time_point: int
                ) -> Dict[int, Position]:
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


def save_tracking_data(experiment: Experiment, output_file: str):
    """Saves the tracking data to the TrackMate format, used by TrackMate, MaMut, Mastodon, LFTree and others."""
    trackmate = Element('TrackMate', {"version": "3.6.0"})
    trackmate.append(Comment("Generated by OrganoidTracker"))

    # Make the model tag
    model = SubElement(trackmate, "Model", {"spatialunits": "µm", "timeunits": "frame"})
    feature_declarations = SubElement(model, "FeatureDeclarations")
    SubElement(feature_declarations, "SpotFeatures")
    SubElement(feature_declarations, "EdgeFeatures")
    SubElement(feature_declarations, "TrackFeatures")

    # Store the positions:
    position_to_id = _store_positions(experiment, model)

    # Store the links
    # We store everything in a single track - this seems to be allowed (and it's also wat LFTree uses)
    all_tracks = SubElement(model, "AllTracks")
    the_track = SubElement(all_tracks, "Track", {"TRACK_ID": str(1), "name": str(1)})
    for source_position, target_position in experiment.links.find_all_links():
        SubElement(the_track, "Edge", {"SPOT_SOURCE_ID": str(position_to_id[source_position]),
                                       "SPOT_TARGET_ID": str(position_to_id[target_position])})

    filtered_tracks = SubElement(model, "FilteredTracks")
    SubElement(filtered_tracks, "TrackID", {"TRACK_ID": str(1)})

    # Make the settings tag
    settings = SubElement(trackmate, "Settings")
    _store_settings(experiment, settings)

    tree = ElementTreeLib.ElementTree(trackmate)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def _store_settings(experiment: Experiment, settings: Element):
    images = experiment.images
    metadata = {"filename": "", "folder": ""}
    image_size_zyx = images.image_loader().get_image_size_zyx()
    if image_size_zyx is not None:
        metadata["nslices"] = str(image_size_zyx[0])
        metadata["height"] = str(image_size_zyx[1])
        metadata["width"] = str(image_size_zyx[2])
    resolution = images.resolution(allow_incomplete=True)
    metadata["voxeldepth"] = str(resolution.pixel_size_z_um)
    metadata["pixelheight"] = str(resolution.pixel_size_y_um)
    metadata["pixelwidth"] = str(resolution.pixel_size_x_um)
    metadata["timeinterval"] = str(resolution.time_point_interval_m)
    if experiment.first_time_point_number() is not None and experiment.last_time_point_number() is not None:
        time_points = experiment.last_time_point_number() - experiment.first_time_point_number() + 1
        metadata["nframes"] = str(time_points)
    SubElement(settings, "ImageData", metadata)
    SubElement(settings, "InitialSpotFilter", {"feature": "QUALITY", "isabove": "true", "value": "0.0"})
    SubElement(settings, "SpotFilterCollection")
    SubElement(settings, "TrackFilterCollection")
    analyzer_collection = SubElement(settings, "AnalyzerCollection")
    SubElement(analyzer_collection, "SpotAnalyzers")
    edge_analyzers = SubElement(analyzer_collection, "EdgeAnalyzers")
    SubElement(edge_analyzers, "Analyzer", {"key": "Edge target"})
    SubElement(analyzer_collection, "TrackAnalyzers")


def _store_positions(experiment: Experiment, model: Element) -> Dict[Position, int]:
    next_id = 0
    position_to_id = dict()
    all_spots = SubElement(model, "AllSpots", {"nspots": str(len(experiment.positions))})
    for time_point in experiment.time_points():
        spots_in_frame = SubElement(all_spots, "SpotsInFrame", {"frame": str(time_point.time_point_number())})
        for position in experiment.positions.of_time_point(time_point):
            position_to_id[position] = next_id

            SubElement(spots_in_frame, "Spot", {"FRAME": str(position.time_point_number()),
                                                "ID": str(next_id),
                                                "POSITION_T": str(float(position.time_point_number())),
                                                "POSITION_X": str(position.x),
                                                "POSITION_Y": str(position.y),
                                                "POSITION_Z": str(position.z),
                                                "QUALITY": "-1",
                                                "RADIUS": "3",
                                                "VISIBILITY": "1",
                                                "NAME": str(next_id)})
            next_id += 1
    return position_to_id
