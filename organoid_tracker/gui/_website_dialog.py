"""Shows a "website" (multiple HTML files, see the Website class) in a popup window. Used by dialog.show_website()."""

import sys
from os import path
from typing import Optional

from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QMainWindow, QWidget, QTextBrowser

from organoid_tracker.core import UserError
from organoid_tracker.gui.website import Website

_MANUALS_FOLDER = "manuals"
_MANUALS_FOLDER_ABSOLUTE = path.join(path.dirname(path.abspath(sys.argv[0])), _MANUALS_FOLDER)
_MAIN_MANUAL = "INDEX.md"
_SCROLL_AREA_PALETTE = QPalette()
_SCROLL_AREA_PALETTE.setColor(QPalette.Background, QtCore.Qt.white)
_DOCUMENT_STYLE = """
font-size: 12pt;
font-family: Georgia, "Times New Roman", serif;
"""


def _file_get_contents(file_name: str):
    with open(file_name, encoding="utf8") as file:
        return file.read()


class _HelpWebsite(Website):

    def get_root_folder(self) -> str:
        return _MANUALS_FOLDER

    def get_title(self) -> str:
        return "Manual"

    def navigate(self, url: str) -> Optional[str]:
        if url == Website.INDEX:
            url = _MAIN_MANUAL

        if ":" in url or ".." in url:
            raise UserError("Unhandled URL", "Don't know how to open " + url)

        file = path.join(_MANUALS_FOLDER_ABSOLUTE, url)
        if not path.isfile(file):
            raise UserError("File not found", url + " does not exist")
        return _file_get_contents(file)


class _HtmlWindow(QMainWindow):

    _text_view: QTextBrowser
    _website: Website

    def __init__(self, parent: QWidget, website: Website):
        super().__init__(parent)
        self._website = website

        self.setMinimumSize(800, 600)
        self.setWindowTitle(website.get_title())

        html = _markdown_to_styled_html(website.navigate(Website.INDEX), relative_to_folder=website.get_root_folder())

        # Setup scrollable layout
        self._text_view = QTextBrowser(self)
        self._text_view.setText(html)
        self._text_view.setStyleSheet(_DOCUMENT_STYLE)
        self._text_view.setOpenLinks(False)
        self._text_view.anchorClicked.connect(self._on_click)

        self.setCentralWidget(self._text_view)

        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def _on_click(self, url: QUrl):
        try:
            markdown = self._website.navigate(url.url())
            if markdown is not None:
                html = _markdown_to_styled_html(markdown, relative_to_folder=self._website.get_root_folder())
                self._text_view.setText(html)
        except BaseException as e:
            from organoid_tracker.gui import dialog
            dialog.popup_exception(e)


def show_help(parent: QWidget):
    _HtmlWindow(parent, _HelpWebsite())


def show_website(parent: QWidget, website: Website):
    _HtmlWindow(parent, website)


def _markdown_to_styled_html(markdown_str: str, *, relative_to_folder: str = ".") -> str:
    """Converts the given Markdown text to HTML, with some basic CSS styles applied."""
    import markdown
    from mdx_gfm import GithubFlavoredMarkdownExtension
    html_text = markdown.markdown(markdown_str,
                      extensions=[GithubFlavoredMarkdownExtension()])

    # Apply special corrections for images
    html_text = html_text.replace("<p><img ", '<p style="line-height: 100; color: #444;'
                                                          ' font-family:sans-serif"><img ')
    html_text = html_text.replace(" src=\"", " src=\"" + relative_to_folder + "/")  # Change images path

    # Construct document
    html_text = """
        <html>
            <head>
                <style type="text/css">
                    code { font-size: 10pt }
                    h1 { margin: 40px 40px 20px 40px }
                    h2 { margin: 40px 40px 20px 40px }
                    h3 { margin: 20px 40px }
                    li { line-height: 140 }
                    p { margin: 20px 40px; line-height: 140 }
                    pre { margin: 40px }
                </style>
            </head>
            <body>
                <div style="font-size:small; color:gray; text-align: center; font-family: sans-serif">OrganoidTracker</div>
                """ + html_text + """
                <div style="height: 40px"></div>
            </body>
        </html>
"""
    return html_text
