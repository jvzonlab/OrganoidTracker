from abc import ABC, ABCMeta
from typing import Optional


class ProgressBar:
    """Abstract class for progress bars. Concrete implementations must implement the methods below."""

    NO_OP: "ProgressBar" = None  # Set at the end of this file

    def set_progress(self, progress: Optional[int]):
        """Sets the progress. 0 is fully empty, 100 is fully done. None is equivalent to calling set_busy()"""
        raise NotImplementedError()

    def set_busy(self):
        """Set the progress bar to a busy state, e.g. by showing an animation. At this point, it is not known how far
        along the task is."""
        raise NotImplementedError()

    def set_error(self):
        """Sets the progress bar to an error state."""
        raise NotImplementedError()


# Set a no-op progress bar, for windows without a progress bar
class _NoOpProgressBar(ProgressBar):
    def set_progress(self, progress: Optional[int]):
        pass

    def set_busy(self):
        pass

    def set_error(self):
        pass


ProgressBar.NO_OP = _NoOpProgressBar()
