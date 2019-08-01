import sys
from os import path

import markdown
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QTextBrowser
from mdx_gfm import GithubFlavoredMarkdownExtension

from ai_track.core import UserError

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


class _HelpFile:

    _html_text: str

    def __init__(self, file_name: str):
        file = path.join(_MANUALS_FOLDER_ABSOLUTE, file_name)
        if not path.isfile(file):
            raise UserError("File not found", file_name + " does not exist")
        file_text = _file_get_contents(file)
        self._html_text = markdown.markdown(file_text,
                          extensions=[GithubFlavoredMarkdownExtension()])

        # Apply a style sheet
        self._html_text = self._html_text.replace("<code", '<code style="font-size: 10pt"')
        self._html_text = self._html_text.replace("<p>", '<p style="line-height: 140; margin: 20px 40px">')
        self._html_text = self._html_text.replace("<pre", '<pre style="margin: 40px"')
        self._html_text = self._html_text.replace("<h1>", '<h1 style="margin: 40px 40px 20px 40px">')
        self._html_text = self._html_text.replace("<h2>", '<h2 style="margin: 40px 40px 20px 40px">')
        self._html_text = self._html_text.replace("<h3>", '<h3 style="margin: 20px 40px">')
        self._html_text = self._html_text.replace("<li>", '<li style="line-height: 140">')
        self._html_text = self._html_text.replace(" src=\"", " src=\"" + _MANUALS_FOLDER + "/")  # Change images path

        # Extra margins at top and bottom:
        self._html_text = '<div style="font-size:small; color:gray; text-align: center; font-family: sans-serif">' \
                          'AI_track manual</div> ' + self._html_text + ' <div style="height: 40px"></div>'
    def html(self):
        return self._html_text


class _HelpWindow(QMainWindow):

    _text_view: QTextBrowser

    def __init__(self, parent: QWidget, help_file: _HelpFile):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Manual")

        # Setup scrollable layout
        self._text_view = QTextBrowser(self)
        self._text_view.setText(help_file.html())
        self._text_view.setStyleSheet(_DOCUMENT_STYLE)
        self._text_view.setOpenLinks(False)
        self._text_view.anchorClicked.connect(self._on_click)

        self.setCentralWidget(self._text_view)

        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def _on_click(self, url: QUrl):
        try:
            help_file = _HelpFile(url.path())
            self._text_view.setText(help_file.html())
        except ValueError as e:
            from ai_track.gui import dialog
            dialog.popup_exception(e)


def show_help(parent: QWidget):
    _HelpWindow(parent, _HelpFile(_MAIN_MANUAL))
