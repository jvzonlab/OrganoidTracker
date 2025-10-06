from os import path

from PySide6.QtGui import QIcon, QPixmap


def _get_icon_file_path(file_name: str) -> str:
    return path.join(path.dirname(path.dirname(path.abspath(__file__))), "icons", file_name)


def get_icon(file_name: str) -> QIcon:
    """Gets an icon from the icons directory. File name should include the file extension, for example
     "file_load.png"."""
    return QIcon(_get_icon_file_path(file_name))


def get_icon_pixmap(file_name: str) -> QPixmap:
    """Gets an icon pixmap from the icons directory. File name should include the file extension, for example
    "file_load.png"."""
    return QPixmap(_get_icon_file_path(file_name))
