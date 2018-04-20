class Parameters:
    """Object used to store measurement parameters."""

    shape_detection_radius: int  # Used for detecting shapes
    intensity_detection_radius: int  # Used for detection (changed) intensities.
    max_distance: int  # Maximum distance between mother and daughter cells
    intensity_detection_radius_large: int

    def __init__(self, **kwargs):
        """Sets all given keyword args as parameters on this object."""
        for name, value in kwargs.items():
            setattr(self, name, value)
