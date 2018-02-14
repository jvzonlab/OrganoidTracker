# Simple image cache


_image_cache = []


def add_to_cache(frame_number: int, image):
    """Adds an image to a cache. The cache removes the oldest element (by addition time) when it becomes too large."""
    global _image_cache
    if len(_image_cache) > 5:
        _image_cache.pop(0)
    _image_cache.append((frame_number, image))


def get_from_cache(frame_number: int):
    global _image_cache
    for entry in _image_cache:
        if entry[0] == frame_number:
            return entry[1]
    return None