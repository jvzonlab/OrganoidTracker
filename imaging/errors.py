from enum import Enum


class Severity(Enum):
    WARNING = 0
    ERROR = 1


NO_FUTURE_POSITION = 1
POTENTIALLY_NOT_A_MOTHER = 2
POTENTIALLY_SHOULD_BE_A_MOTHER = 3
TOO_MANY_DAUGHTER_CELLS = 4
NO_PAST_POSITION = 5
CELL_MERGE = 6


__info = {
    NO_FUTURE_POSITION: (Severity.WARNING, "No future position for cell; dead cell?"),
    POTENTIALLY_NOT_A_MOTHER: (Severity.WARNING, "Maybe not a mother"),
    POTENTIALLY_SHOULD_BE_A_MOTHER: (Severity.WARNING, "Should maybe be a mother"),
    TOO_MANY_DAUGHTER_CELLS: (Severity.ERROR, "Has more than two daughter cells"),
    NO_PAST_POSITION: (Severity.ERROR, "Cell popped up out of nothing"),
    CELL_MERGE: (Severity.ERROR, "Two cells merged together into this cell")
}


def get_severity(error: int) -> Severity:
    """Gets the severity of the given error."""
    try:
        return __info[error][0]
    except KeyError:
        return Severity.ERROR


def get_message(error: int) -> str:
    """Gets the error message corresponding to the given error."""
    try:
        return __info[error][1]
    except KeyError:
        return "Unknown error code " + str(error)


