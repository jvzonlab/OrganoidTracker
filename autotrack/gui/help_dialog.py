import sys
from os import path

import markdown
from PyQt5 import QtCore
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QScrollArea
from mdx_gfm import GithubFlavoredMarkdownExtension

_MANUALS_FOLDER = path.join(path.dirname(path.abspath(sys.argv[0])), 'manuals')
_MAIN_MANUAL = "VISUALIZER.md"
_SCROLL_AREA_PALETTE = QPalette()
_SCROLL_AREA_PALETTE.setColor(QPalette.Background, QtCore.Qt.white)
_DOCUMENT_STYLE = """
font-size: 12pt;
font-family: Georgia, "Times New Roman", serif;
padding: 12px;
background-color: white;
"""


def link_handler(arg):
    print("Hi1" + arg)


def _file_get_contents(file_name: str):
    with open(file_name) as file:
        return file.read()


class _HelpFile:

    _html_text: str

    def __init__(self, file_name: str):
        file_text = _file_get_contents(path.join(_MANUALS_FOLDER, file_name))
        self._html_text = markdown.markdown(file_text,
                          extensions=[GithubFlavoredMarkdownExtension()])
        self._html_text = self._html_text.replace("<code", '<code style="font-size: 10pt"')
        self._html_text = self._html_text.replace("<p>", '<p style="line-height: 140">')

    def html(self):
        print(self._html_text)
        return self._html_text


class _HelpWindow(QMainWindow):

    def __init__(self, parent: QWidget, help_file: _HelpFile):
        super().__init__(parent)
        self.setMinimumSize(750, 600)
        self.setWindowTitle("Manual")

        # Setup scrollable layout
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        scroll_area = QScrollArea(central_widget)
        scroll_area.setAutoFillBackground(True)
        scroll_area.setPalette(_SCROLL_AREA_PALETTE)
        layout.addWidget(scroll_area)

        # Create main text widget
        html_text = QLabel(scroll_area)
        html_text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        html_text.setText(help_file.html())
        html_text.setStyleSheet(_DOCUMENT_STYLE)
        html_text.setWordWrap(True)
        html_text.setOpenExternalLinks(False)
        html_text.setFixedWidth(700)
        html_text.linkActivated.connect(link_handler)
        scroll_area.setWidget(html_text)

        self.setCentralWidget(central_widget)

        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)


def show_help(parent: QWidget):
    _HelpWindow(parent, _HelpFile(_MAIN_MANUAL))
