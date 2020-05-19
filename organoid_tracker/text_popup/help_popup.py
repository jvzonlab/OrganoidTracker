from typing import Optional

from organoid_tracker.core import UserError
from organoid_tracker.text_popup.text_popup import RichTextPopup
from os import path


_MANUALS_FOLDER = "manuals"
_MANUALS_FOLDER_ABSOLUTE = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), _MANUALS_FOLDER)
_MAIN_MANUAL = "INDEX.md"


def _file_get_contents(file_name: str):
    with open(file_name, encoding="utf8") as file:
        return file.read()


class HelpPopup(RichTextPopup):

    def get_root_folder(self) -> str:
        return _MANUALS_FOLDER

    def get_title(self) -> str:
        return "Manual"

    def navigate(self, url: str) -> Optional[str]:
        if url == RichTextPopup.INDEX:
            url = _MAIN_MANUAL

        if ":" in url or ".." in url:
            raise UserError("Unhandled URL", "Don't know how to open " + url)

        file = path.join(_MANUALS_FOLDER_ABSOLUTE, url)
        if not path.isfile(file):
            raise UserError("File not found", url + " does not exist")
        return _file_get_contents(file)