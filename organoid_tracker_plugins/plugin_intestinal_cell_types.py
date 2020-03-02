"""Registers the Stem and Paneth cell type."""
from typing import Optional, Any, Dict

from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.spline import Spline
from organoid_tracker.core.position import Position
from organoid_tracker.core.marker import Marker
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers

# Stem cells
STEM_PRECURSOR = Marker([Position], "STEM_PRECURSOR", "precursor of a stem cell", (255, 4, 0))
STEM_PROGENY = Marker([Position], "STEM_PROGENY", "progeny of a (supposed) stem cell", (255, 100, 100))
STEM = Marker([Position], "STEM", "stem cell", (182, 1, 1))

# Absorptive
ABSORPTIVE_PRECURSOR = Marker([Position], "ABSORPTIVE_PRECURSOR", "precursor of an absorptive cell", (0, 8, 255))
ABSORPTIVE_PROGENY = Marker([Position], "ABSORPTIVE_PROGENY", "progeny of a (supposedly) absorptive cell", (0, 8, 255))
M_CELL = Marker([Position], "M_CELL", "M cell", (0, 165, 255))
ENTEROCYTE = Marker([Position], "ENTEROCYTE", "enterocyte cell", (16, 0, 105))

# Secretive
SECRETIVE_PRECURSOR = Marker([Position], "SECRETIVE_PRECURSOR", "precursor of a secretory cell", (255, 250, 0))
SECRETIVE_PROGENY = Marker([Position], "SECRETIVE_PROGENY", "progeny of a (supposedly) secretory cell", (255, 250, 0))
PANETH = Marker([Position], "PANETH", "Paneth cell", (0, 255, 12))
GOBLET = Marker([Position], "GOBLET", "Goblet cell", (95, 173, 0))
ENTEROENDOCRINE = Marker([Position], "ENTEROENDOCRINE", "enteroendocrine cell", (255, 170, 0))
TUFT = Marker([Position], "TUFT", "Tuft cell", (255, 136, 120))


# Markers that mark something else
LUMEN = Marker([Position], "LUMEN", "lumen", (200, 200, 200))
CRYPT = Marker([Spline], "CRYPT", "crypt axis", (255, 0, 0), is_axis=True)


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    for marker in [STEM_PRECURSOR, STEM_PROGENY, STEM, ABSORPTIVE_PRECURSOR, ABSORPTIVE_PROGENY, M_CELL, ENTEROCYTE,
                   SECRETIVE_PRECURSOR, SECRETIVE_PROGENY, PANETH, GOBLET, ENTEROENDOCRINE, TUFT, LUMEN]:
        gui_experiment.register_marker(marker)

    gui_experiment.register_marker(CRYPT)


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Types-Assign parent cell types...": lambda: _assign_types(window)
    }


def _assign_types(window: Window):
    if not dialog.prompt_confirmation("Staining types", "This assigns types to mother cells based on the type of the"
                                      " daughter cell. For example, the parent of a stem cell also becomes a stem cell."
                                      "\n\nYou cannot undo this operation. Do you want to continue?"):
        return

    # Loop through all ending tracks to add precursor types
    experiment = window.get_experiment()
    links = experiment.links
    position_data = experiment.position_data
    for track in links.find_all_tracks():
        if len(track.get_next_tracks()) > 0:
            continue  # Not an ending track

        _assign_precursor_type_recursive(position_data, track)

    # Loop through all divisions to add progeny types
    for starting_track in links.find_starting_tracks():
        for track in starting_track.find_all_descending_tracks(include_self=True):
            # This way of iterating (starting tracks, then descending) ensures that were are going forward in time

            next_tracks = track.get_next_tracks()
            if len(next_tracks) != 2:
                continue
            next_track_1, next_track_2 = next_tracks
            next_type_1 = linking_markers.get_position_type(position_data, next_track_1.find_first_position())
            next_type_2 = linking_markers.get_position_type(position_data, next_track_2.find_first_position())
            mother_type = linking_markers.get_position_type(position_data, track.find_last_position())
            if next_type_1 is None:
                _assign_position_type_if_not_none(position_data, next_track_1, _get_progeny_type(mother_type, next_type_2))
            if next_type_2 is None:
                _assign_position_type_if_not_none(position_data, next_track_2, _get_progeny_type(mother_type, next_type_1))

    window.get_gui_experiment().redraw_data()


def _assign_precursor_type_recursive(position_data: PositionData, track: LinkingTrack):
    """Assigns a precursor type to all parents of this track, all the way back to the first time point. If the given
    track is not a daughter track, this method does nothing."""
    previous_tracks = track.get_previous_tracks()
    if len(previous_tracks) != 1:
        return  # Need to have one parent cell

    parent_track = previous_tracks.pop()
    if _is_known_type(linking_markers.get_position_type(position_data, parent_track.find_last_position())):
        return  # Don't overwrite known types - we only want to overwrite guessed precursor types and

    sibling_tracks = parent_track.get_next_tracks()
    if len(sibling_tracks) != 2:
        return  # Need to have two daughter cells

    # Get precursor type
    daughter1_type = linking_markers.get_position_type(position_data, sibling_tracks.pop().find_first_position())
    daughter2_type = linking_markers.get_position_type(position_data, sibling_tracks.pop().find_first_position())
    precursor_type = _get_precursor_type(daughter1_type, daughter2_type)

    # Apply precursor type
    _assign_position_type_if_not_none(position_data, parent_track, precursor_type)

    # Go back in time
    _assign_precursor_type_recursive(position_data, parent_track)


def _assign_position_type_if_not_none(position_data: PositionData, track: LinkingTrack, position_type: str):
    if position_type is None:
        return

    for position in track.positions():
        linking_markers.set_position_type(position_data, position, position_type)


def _is_known_type(type_name: Optional[str]) -> bool:
    """Returns True if this type is of a "sure" type, for example from antibody staining or live markers."""
    if type_name is None:
        return False
    if type_name.endswith("_PROGENY") or type_name.endswith("_PRECURSOR"):
        return False
    return True


def _is_stem(type_name: Optional[str]) -> bool:
    """Returns true if the cell is a known stem cell or a precursor of a known stem cell.

    Note: this method returns False for STEM_PROGENY, as that can be any cell type."""
    return type_name in {STEM_PRECURSOR.save_name, STEM.save_name}


def _is_absorptive(type_name: Optional[str]) -> bool:
    """Gets whether the cell type is an absorptive cell type, a precursor or a progeny.."""
    return type_name in {ABSORPTIVE_PROGENY.save_name, ABSORPTIVE_PRECURSOR.save_name, M_CELL.save_name,
                         ENTEROCYTE.save_name}


def _is_secretive(type_name: Optional[str]) -> bool:
    """Gets whether the cell type is an absorptive cell type, a precursor or a progeny."""
    return type_name in {SECRETIVE_PROGENY.save_name, SECRETIVE_PRECURSOR.save_name, PANETH.save_name, GOBLET.save_name,
                         ENTEROENDOCRINE.save_name, TUFT.save_name}


def _get_precursor_type(daughter1_type: Optional[str], daughter2_type: Optional[str]) -> Optional[str]:
    # Assume stem cells come from stem cells
    if _is_stem(daughter1_type) or _is_stem(daughter2_type):
        return STEM_PRECURSOR.save_name

    absorptive1 = _is_absorptive(daughter1_type)
    absorptive2 = _is_absorptive(daughter2_type)
    secretive1 = _is_secretive(daughter1_type)
    secretive2 = _is_secretive(daughter2_type)
    sure1 = _is_known_type(daughter1_type)
    sure2 = _is_known_type(daughter2_type)

    if absorptive1 == absorptive2 == True\
            or (absorptive1 and sure1 and daughter2_type is None)\
            or (daughter1_type is None and absorptive2 and sure2):
        # If both are supposedly absorptive, or one is definitely absorptive and the other is unknown, assume
        # absorptive precursor
        return ABSORPTIVE_PRECURSOR.save_name
    if secretive1 == secretive2 == True\
            or (secretive1 and sure1 and daughter2_type is None) \
            or (daughter1_type is None and secretive2 and sure2):
        # If both are supposedly secretive, or one is definitely secretive and the other is unknown, assume secretive
        # precursor
        return SECRETIVE_PRECURSOR.save_name

    return None


def _get_progeny_type(mother_type: Optional[str], sibling_type: Optional[str]) -> Optional[str]:
    if mother_type is None:
        # Precursor types must already be known. If no precursor type is available, then we also don't know what the
        # progeny type is
        return None

    if _is_stem(mother_type):
        return STEM_PROGENY.save_name

    if _is_absorptive(mother_type):
        return ABSORPTIVE_PROGENY.save_name

    if _is_secretive(mother_type):
        return SECRETIVE_PROGENY.save_name

    return None
