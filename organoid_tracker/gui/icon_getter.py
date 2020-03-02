import sys
from os import path

from PyQt5.QtGui import QIcon


def get_icon(file_name: str) -> QIcon:
    return QIcon(path.join(path.dirname(path.abspath(sys.argv[0])), 'organoid_tracker', 'icons', file_name))
