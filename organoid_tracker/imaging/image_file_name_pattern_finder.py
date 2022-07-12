import re
from typing import Optional


def find_time_and_channel_pattern(file_name: str) -> Optional[str]:
    """Given a file name like "image_t1_c1.png", returns "image_t{time}_c{channel}.png". Returns None if no pattern
    (time or channel) was found."""
    time_pattern = _find_time_pattern(file_name)
    if time_pattern is None:
        channel_pattern = _find_channel_pattern(file_name)
        if channel_pattern is None:
            return None  # Found no patterns
        return channel_pattern  # No time pattern, at least we could find a channel pattern
    channel_pattern = _find_channel_pattern(time_pattern)
    if channel_pattern is None:
        return time_pattern  # At least we could find a time pattern
    return channel_pattern  # Found both patterns


def _fixup_pattern(label: str, pattern: str) -> str:
    """Replaces the useless {label:01} with just {label}."""
    return pattern.replace("{" + label + ":01}", "{" + label + "}")


def _find_time_pattern(file_name: str) -> Optional[str]:
    """Given a file name like "image_1.png", returns "image_{time}.png". Returns None if no pattern for the time can
    be found."""
    # Support t001
    counting_part = re.search('t0*[01]', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return _fixup_pattern("time", file_name[0:start] + "t{time:0" + str(end - start - 1) + "}" + file_name[end:])

    # Support T001
    counting_part = re.search('T0*[01]', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return _fixup_pattern("time", file_name[0:start] + "T{time:0" + str(end - start - 1) + "}" + file_name[end:])

    # Support 001.extension
    counting_part = re.search('0*[01]\\.[A-Za-z]', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        prefix = file_name[0:start]
        if prefix.endswith("position") or prefix.endswith("xy") or prefix.endswith("pos"):
            # If the file ends with "position001.tif" (or similar),
            # then "001" is not a time pattern
            return None
        return _fixup_pattern("time", prefix + "{time:0" + str(end - start - 2) + "}." + file_name[end-1:])

    # Support (001).extension
    counting_part = re.search('\\(0*[01]\\)\\.[A-Za-z]', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return _fixup_pattern("time", file_name[0:start] + "({time:0" + str(end - start - 4) + "})." + file_name[end-1:])

    # Support z001_ (worm microscope of Zon lab)
    counting_part = re.search('^z(0*[01])_', file_name)
    if counting_part is not None:
        start, end = counting_part.start(1), counting_part.end(1)
        return _fixup_pattern("time",
                              file_name[0:start] + "{time:0" + str(end - start) + "}" + file_name[end:])

    # Fail
    return None


def _find_channel_pattern(file_name: str) -> Optional[str]:
    """Given a file name like "image_t1_c1.png", returns "image_t1_c{channel}.png". Returns None if no pattern for the time can
    be found."""
    # Support c001
    counting_part = re.search('c0*[01]', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return _fixup_pattern("channel", file_name[0:start] + "c{channel:0" + str(end - start - 1) + "}" + file_name[end:])

    # Support C001
    counting_part = re.search('C0*[01]', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return _fixup_pattern("channel", file_name[0:start] + "C{channel:0" + str(end - start - 1) + "}" + file_name[end:])

    # Fail
    return None
