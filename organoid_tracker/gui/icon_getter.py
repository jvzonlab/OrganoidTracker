import sys
from os import path

from PySide6.QtGui import QIcon


def get_icon(file_name: str) -> QIcon:
    return QIcon(path.join(path.dirname(path.dirname(path.abspath(__file__))), "icons", file_name))
