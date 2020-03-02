from typing import Set, Hashable
from threading import Lock


class ConcurrentSet:
    """A set that uses locks everywhere to make it suitable for concurrent use. No iterator support yet."""
    _internal: Set
    _lock: Lock

    def __init__(self):
        self._internal = set()
        self._lock = Lock()

    def add(self, object: Hashable):
        """
        Add an element to a set.

        This has no effect if the element is already present.
        """
        with self._lock:
            self._internal.add(object)

    def remove(self, object: Hashable):
        """
        Remove an element from a set; it must be a member.

        If the element is not a member, raise a KeyError.
        """
        with self._lock:
            self._internal.remove(object)

    def __len__(self) -> int:
        """ Return the number of items in a container. """
        with self._lock:
            return len(self._internal)

    def __contains__(self, item):
        with self._lock:
            return item in self._internal
