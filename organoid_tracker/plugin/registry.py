from typing import Dict, Optional, Iterable, Type

from organoid_tracker.core.marker import Marker


class Registry:
    """A plugin-controlled registry for various plugin-provided things, like cell types.

    For example, to add something to the marker list, plugins can define a get_markers() method.
    See the plugin documentation for more information
    """

    _markers: Dict[str, Marker]

    def __init__(self):
        self._markers = dict()

    def clear_all(self):
        """Clears the registry."""
        self._markers.clear()

    def get_marker_by_save_name(self, save_name: Optional[str]) -> Optional[Marker]:
        """Gets the marker using the given save name. Returns None if no marker exists for that save name."""
        if save_name is None:
            return None
        return self._markers.get(save_name)

    def get_registered_markers(self, type: Type) -> Iterable[Marker]:
        """Gets all registered markers for the given type. For example, you can ask all registered cell types using
        get_registered_markers(Position)."""
        for marker in self._markers.values():
            if marker.applies_to(type):
                yield marker
