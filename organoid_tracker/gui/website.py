from abc import ABC
from typing import Optional


class Website(ABC):
    """Represents a "website" (a collection of HTML pages), shown using dialog.show_website(). The help files are one
    example of a Website. If you need to build a quick GUI with text, """

    INDEX = "index"  # The first page that is loaded

    def get_root_folder(self) -> str:
        """Paths for things like images will be relative to this path."""
        return "."

    def get_title(self) -> str:
        """Returns the title for the whole website."""
        raise NotImplementedError()

    def navigate(self, url: str) -> Optional[str]:
        """Returns the Markdown for the given URL. The website can also perform some other action, like zooming in on a
        position or starting a program. If this method returns None, then the navigation is cancelled."""
        raise NotImplementedError()
