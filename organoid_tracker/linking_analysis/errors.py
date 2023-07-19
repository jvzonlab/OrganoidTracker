"""Definition of all available errors. See linking_markers for how to apply these errors."""

from enum import Enum


class Severity(Enum):
    WARNING = 0
    ERROR = 1


class Error(Enum):
    NO_FUTURE_POSITION = 1
    POTENTIALLY_NOT_A_MOTHER = 2  # Unused in OrganoidTracker 2
    POTENTIALLY_SHOULD_BE_A_MOTHER = 3
    TOO_MANY_DAUGHTER_CELLS = 4
    NO_PAST_POSITION = 5
    CELL_MERGE = 6
    POTENTIALLY_WRONG_DAUGHTERS = 7  # Unused in OrganoidTracker 2
    YOUNG_MOTHER = 8
    LOW_MOTHER_SCORE = 9
    SHRUNK_A_LOT = 10
    MOVED_TOO_FAST = 11
    FAILED_SHAPE = 12  # Unused in OrganoidTracker 2
    UNCERTAIN_POSITION = 13
    LOW_LINK_SCORE = 14

    def get_severity(self) -> Severity:
        """Gets the severity."""
        return _get_severity(self)

    def get_message(self) -> str:
        """Gets an user-friendly error message."""
        return _get_message(self)


__info = {
    Error.NO_FUTURE_POSITION: (Severity.WARNING, "This cell has no links to the future. Please check if this is correct."),
    Error.POTENTIALLY_NOT_A_MOTHER: (Severity.WARNING, "This division is maybe wrong; nearby cell has similar likeliness."),
    Error.POTENTIALLY_SHOULD_BE_A_MOTHER: (Severity.WARNING, "A division was probably missed here; found a high division score."),
    Error.TOO_MANY_DAUGHTER_CELLS: (Severity.ERROR, "This cell has more than two daughter cells."),
    Error.NO_PAST_POSITION: (Severity.ERROR, "This cell popped up out of nothing."),
    Error.CELL_MERGE: (Severity.ERROR, "Two cells merged together into this cell."),
    Error.POTENTIALLY_WRONG_DAUGHTERS: (Severity.WARNING, "One of the two daughter cells is maybe wrong."),
    Error.YOUNG_MOTHER: (Severity.ERROR, "This division appeared very quickly after the previous. One of those is likely wrong."),
    Error.LOW_MOTHER_SCORE: (Severity.WARNING, "This division is maybe wrong; division score is low."),
    Error.SHRUNK_A_LOT: (Severity.WARNING, "This cell just shrank a lot in size. Maybe a division was missed, or the shape detection failed."),
    Error.MOVED_TOO_FAST: (Severity.WARNING, "This cell just moved very quickly. The link coming from the past may be"
                                             " wrong."),
    Error.FAILED_SHAPE: (Severity.WARNING, "This cell has an irregular shape. Maybe it was a misdetection, or it should be"
                                       " a mother cell."),
    Error.UNCERTAIN_POSITION: (Severity.WARNING, "Uncertain if there actually is a cell here."),
    Error.LOW_LINK_SCORE: (Severity.WARNING, "This is probably not a correct link; the link score is low.")
}


def _get_severity(error: Error) -> Severity:
    """Gets the severity of the given error."""
    try:
        return __info[error][0]
    except KeyError:
        return Severity.ERROR


def _get_message(error: Error) -> str:
    """Gets the error message corresponding to the given error."""
    try:
        return __info[error][1]
    except KeyError:
        return "Unknown error code " + str(error)
