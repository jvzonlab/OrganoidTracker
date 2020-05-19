from abc import ABC
from typing import Optional


class RichTextPopup(ABC):
    """Represents a "rich text popup" (a collection of HTML pages), shown using dialog.popup_rich_text(). The help files
    are one example of a RichTextPopup. If you need to build a quick GUI with text, this class could help you."""

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
