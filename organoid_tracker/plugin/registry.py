from typing import Dict, Optional, Iterable, Type, List

from organoid_tracker.core.marker import Marker
from organoid_tracker.imaging.file_loader import FileLoader


class Registry:
    """A plugin-controlled registry for various plugin-provided things, like cell types or file handlers.

    For example, to add something to the marker list, plugins can define a get_markers() method in their main module.
    See the plugin documentation for more information.
    """

    _markers: Dict[str, Marker]
    _file_loaders: List[FileLoader]

    def __init__(self):
        self._markers = dict()
        self._file_loaders = list()

    def clear_all(self):
        """Clears the registry."""
        self._markers.clear()
        self._file_loaders.clear()

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

    def get_registered_file_loaders(self) -> Iterable[FileLoader]:
        """Gets all registered file handlers."""
        yield from self._file_loaders
