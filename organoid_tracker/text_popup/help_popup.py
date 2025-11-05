import webbrowser
from os import path
from typing import Optional

from organoid_tracker.core import UserError
from organoid_tracker.text_popup.text_popup import RichTextPopup

_MANUALS_FOLDER = "manuals"
_MANUALS_FOLDER_ABSOLUTE = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), _MANUALS_FOLDER)
_MAIN_MANUAL = "index.md"


def _file_get_contents(file_name: str):
    with open(file_name, encoding="utf8") as file:
        return file.read()


class HelpPopup(RichTextPopup):

    def get_root_folder(self) -> str:
        return _MANUALS_FOLDER

    def get_title(self) -> str:
        return "Manual"

    def navigate(self, url: str) -> Optional[str]:
        # Open WWW links in a web browser
        if url.startswith("https:") or url.startswith("http:"):
            webbrowser.open(url)
            return None

        if url == RichTextPopup.INDEX:
            url = _MAIN_MANUAL

        if ":" in url or ".." in url:
            raise UserError("Unhandled URL", "Don't know how to open " + url)

        file = path.join(_MANUALS_FOLDER_ABSOLUTE, url)
        if not path.isfile(file):
            raise UserError("File not found", url + " does not exist")
        markdown_str =  _file_get_contents(file)

        # Replace :::{note} style blocks with a block using horizontal lines
        markdown_str = markdown_str.replace(":::{note}", "\n----------------\n### Note:")
        markdown_str = markdown_str.replace(":::", "\n-----------------\n")

        # Cut off everything after :::{eval-rst}  (that is metadata for the Sphinx documentation builder)
        try:
            remove_start_index = markdown_str.index(":::{eval-rst}")
            markdown_str = markdown_str[0:remove_start_index]
        except ValueError:
            pass  # There's no metadata

        return markdown_str
