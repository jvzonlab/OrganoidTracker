"""Shows a "website" (multiple HTML files, see the Website class) in a popup window. Used by dialog.show_website()."""

from os import path

from PySide2 import QtCore
from PySide2.QtCore import QUrl
from PySide2.QtGui import QPalette
from PySide2.QtWidgets import QMainWindow, QWidget, QTextBrowser

from organoid_tracker.text_popup.text_popup import RichTextPopup

_SCROLL_AREA_PALETTE = QPalette()
_SCROLL_AREA_PALETTE.setColor(QPalette.Background, QtCore.Qt.white)
_DOCUMENT_STYLE = """
font-size: 12pt;
"""


class _HtmlWindow(QMainWindow):

    _text_view: QTextBrowser
    _website: RichTextPopup

    def __init__(self, parent: QWidget, website: RichTextPopup):
        super().__init__(parent)
        self._website = website

        self.setMinimumSize(800, 600)
        self.setWindowTitle(website.get_title())

        html = _markdown_to_styled_html(website.navigate(RichTextPopup.INDEX), relative_to_folder=website.get_root_folder())

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


def show_popup(parent: QWidget, website: RichTextPopup):
    _HtmlWindow(parent, website)


def _markdown_to_styled_html(markdown_str: str, *, relative_to_folder: str = ".") -> str:
    """Converts the given Markdown text to HTML, with some basic CSS styles applied."""
    import markdown
    from markdown.extensions.fenced_code import FencedCodeExtension
    html_text = markdown.markdown(markdown_str, extensions=[FencedCodeExtension()])

    # Apply special corrections for images
    html_text = html_text.replace("<p><img ", '<p style="line-height: 1; color: #444;'
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
                    li { line-height: 1.4 }
                    p { margin: 20px 40px; line-height: 1.4 }
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
