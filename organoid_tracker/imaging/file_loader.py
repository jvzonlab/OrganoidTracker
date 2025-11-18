"""An abstract file loader that can open certain file types (images, tracking data) and load them into an experiment.
The program ships with several built-in file loaders, which just wrap functions like
`organoid_tracker.imaging.io.load_data_file(...)`. It's fine to call those functions directly. But using this system,
the GUI can track which file formats are supported (for the dialogs or drag-and-drop), and plugins can add support for
additional file formats.
"""

from abc import abstractmethod, ABC
from enum import Enum, auto
from typing import List

from organoid_tracker.core.experiment import Experiment


class LoadInto:
    """When loading a file, there will always be one open experiment already. In addition, you can create extra
    experiments, which will be shown in extra tabs added to the GUI."""

    experiment: Experiment
    allow_extra_tabs: bool = True  # Set this to False to prevent adding extra tabs
    extra_experiments: List[Experiment]

    _returned_first: bool = False

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.extra_experiments = []

    def next_experiment(self) -> Experiment:
        """First time this method is called, it returns the main experiment. Subsequent calls create new experiments,
        add those to the self.extra_tabs list, and then return them."""
        if not self._returned_first:
            self._returned_first = True
            return self.experiment
        else:
            new_experiment = Experiment()
            self.extra_experiments.append(new_experiment)
            return new_experiment


class FileLoaderType(Enum):
    """The type of file that a FileHandler can open.

    This influences a few things in the GUI:
    - In which loading button the handler shows up (image loading button vs tracking data button).
    - For tracking data, a warning is shown first if there were any unsaved changes.

    When you're loading a file format that contains both images and tracking data, prefer TRACKING. Alternatively,
    you can register two loaders, one that only loads the image data from the file, and one that only loads the
    tracking data.
    """
    IMAGE = auto()
    TRACKING = auto()


class FileLoader(ABC):
    """A file handler that can open certain file types (images, tracking data) and load them into an experiment."""

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of this file handler, for example "TIFF image sequence"."""
        raise NotImplementedError()

    @abstractmethod
    def get_file_patterns(self) -> set[str]:
        """Returns a set of file patterns (like "*.tif") that this handler can open. Patterns should be in lower case.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_file_interactive(self, file_path: str, *, into: LoadInto) -> bool:
        """Loads the file at the given path and adds its contents into the given experiment. This method is allowed
        to interact with the user, for example by showing a dialog to ask for additional information.

        If a file loader wants to add extra tabs to the GUI (for example, when loading multiple image sequences),
        those can be added to the `create_extra_tabs` list. This list should start empty, and any Experiment instances
        added to it will be shown in extra tabs.

        Returns whether loading was successful. If loading was cancelled or failed, return False.

        Note: if you're replacing the image loader, make sure to call experiment.images.close_image_loader() first,
        to avoid a memory and file handler leak.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_type(self) -> FileLoaderType:
        """Returns the type of file that this handler can open."""
        raise NotImplementedError()
