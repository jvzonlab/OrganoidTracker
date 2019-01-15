from typing import Optional, List

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QGroupBox, QFormLayout, QLabel, QComboBox, QDialogButtonBox, QPushButton, \
    QVBoxLayout


class _ImageSeriesSelectionWindow(QDialog):

    image_series_selector: QComboBox

    def __init__(self, title: str, header: str, label: str, options: List[str]):
        super().__init__()

        self.setWindowTitle(title)
        form_box = QGroupBox(header, parent=self)
        form = QFormLayout()

        self.image_series_selector = QComboBox(self)
        for index, name in enumerate(options):
            self.image_series_selector.addItem(name, index)
        form.addRow(QLabel(label, parent=form_box), self.image_series_selector)
        form_box.setLayout(form)

        button_box = QDialogButtonBox(parent=self)
        button_box.addButton(QDialogButtonBox.Ok)
        button_box.addButton(QDialogButtonBox.Cancel)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(form_box)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setVisible(True)


def popup_image_getter(title: str, header: str, label: str, options: List[str]) -> Optional[int]:
    """Shows a popup that allows you to choose from the given list of options. If `len(options) == 1`, then this method
    doesn't show a popup, but immediately returns 0. If options is empty, None is returned."""
    if len(options) == 0:
        return None
    if len(options) == 1:
        return 0

    popup = _ImageSeriesSelectionWindow(title, header, label, options)
    result = popup.exec_()
    if result != QDialog.Accepted:
        return None
    return popup.image_series_selector.currentIndex()
