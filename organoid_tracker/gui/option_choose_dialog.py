from typing import Optional, List

from PySide2 import QtCore
from PySide2.QtCore import Qt, QEvent, qApp
from PySide2.QtGui import QFontMetrics, QStandardItem, QPalette
from PySide2.QtWidgets import QDialog, QGroupBox, QFormLayout, QLabel, QComboBox, QDialogButtonBox, QVBoxLayout, \
    QStyledItemDelegate


class MultipleComboBox(QComboBox):
    """Combobox that allows multitple items to be selected. Source code from
    https://gis.stackexchange.com/a/351152"""

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(MultipleComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text: str, data: Optional[int] =None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts: List[str], datalist: Optional[List[int]]=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self) -> List[int]:
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res


class _ListItemSelectionWindow(QDialog):
    """Window that shows a QComboBox and some text"""

    image_series_selector: QComboBox

    def __init__(self, title: str, header: str, label: str, options: List[str], *, allow_multiple: bool):
        super().__init__()

        self.setWindowTitle(title)
        form_box = QGroupBox(header, parent=self)
        form = QFormLayout()

        self.image_series_selector = MultipleComboBox(self) if allow_multiple else QComboBox(self)
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


def prompt_list(title: str, header: str, label: str, options: List[str]) -> Optional[int]:
    """Shows a popup that allows you to choose from the given list of options. If `len(options) == 1`, then this method
    doesn't show a popup, but immediately returns 0. If options is empty, None is returned."""
    if len(options) == 0:
        return None
    if len(options) == 1:
        return 0

    popup = _ListItemSelectionWindow(title, header, label, options, allow_multiple=False)
    result = popup.exec_()
    if result != QDialog.Accepted:
        return None
    return popup.image_series_selector.currentIndex()


def prompt_list_multiple(title: str, header: str, label: str, options: List[str]) -> Optional[List[int]]:
    """Shows a popup that allows you to choose zero or more options from the given list of options. If options is
    empty, [] is returned. Returns None if cancelled."""
    if len(options) == 0:
        return []

    popup = _ListItemSelectionWindow(title, header, label, options, allow_multiple=True)
    result = popup.exec_()
    if result != QDialog.Accepted:
        return None
    return popup.image_series_selector.currentData()
