"""A custom popup for setting the image resolution in an experiment."""

from typing import Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDoubleSpinBox, QDialog, QPushButton, QWidget, QGroupBox, QFormLayout, QLabel, \
    QDialogButtonBox, QVBoxLayout

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.resolution import ImageResolution
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.undo_redo import UndoableAction
from autotrack.gui.window import Window


class _ResolutionEditorWindow(QDialog):

    x_res: QDoubleSpinBox
    y_res: QDoubleSpinBox
    z_res: QDoubleSpinBox
    t_res: QDoubleSpinBox
    ok_button: QPushButton

    def __init__(self, image_resolution: Optional[ImageResolution]):
        super().__init__()

        self.setWindowTitle("Image resolution")

        form_box = QGroupBox("Image resolution", parent=self)
        form = QFormLayout()
        self.x_res = QDoubleSpinBox(parent=form_box)
        self.x_res.setSuffix("   μm/px")
        self.x_res.setValue(0 if image_resolution is None else image_resolution.pixel_size_x_um)
        self.x_res.valueChanged.connect(self._on_field_change)
        form.addRow(QLabel("Horizontal resolution:", parent=form_box), self.x_res)
        self.y_res = QDoubleSpinBox(parent=form_box)
        self.y_res.setSuffix("   μm/px")
        self.y_res.setValue(0 if image_resolution is None else image_resolution.pixel_size_x_um)
        self.y_res.setEnabled(False)
        form.addRow(QLabel("Vertical resolution: (equal to horizontal)", parent=form_box), self.y_res)
        self.z_res = QDoubleSpinBox(parent=form_box)
        self.z_res.setSuffix("   μm/px")
        self.z_res.setValue(0 if image_resolution is None else image_resolution.pixel_size_z_um)
        self.z_res.valueChanged.connect(self._on_field_change)
        form.addRow(QLabel("Z-resolution:", parent=form_box), self.z_res)
        self.t_res = QDoubleSpinBox(parent=form_box)
        self.t_res.setSuffix("   min/tp")
        self.t_res.setValue(0 if image_resolution is None else image_resolution.time_point_interval_m)
        self.t_res.valueChanged.connect(self._on_field_change)
        form.addRow(QLabel("Time resolution:", parent=form_box), self.t_res)
        form_box.setLayout(form)

        button_box = QDialogButtonBox(parent=self)
        self.ok_button = QPushButton("OK", parent=button_box)
        button_box.addButton(self.ok_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(QDialogButtonBox.Cancel)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(form_box)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setVisible(True)

        self._on_field_change()  # Validate initial values

    def _on_field_change(self):
        self.y_res.setValue(self.x_res.value())

        valid = True
        if self.x_res.value() <= 0:
            valid = False
        if self.z_res.value() <= 0:
            valid = False
        if self.t_res.value() <= 0:
            valid = False
        self.ok_button.setEnabled(valid)


class _UpdateImageResolutionAction(UndoableAction):
    """Action to update the image resolution in an experiment."""
    _new_resolution: ImageResolution
    _old_resolution: Optional[ImageResolution]

    def __init__(self, old_resolution: ImageResolution, new_resolution: ImageResolution):
        self._new_resolution = new_resolution
        self._old_resolution = old_resolution

    def do(self, experiment: Experiment) -> str:
        experiment.images.set_resolution(self._new_resolution)
        return "Updated image resolution"

    def undo(self, experiment: Experiment) -> str:
        experiment.images.set_resolution(self._old_resolution)
        return "Restored image resolution"


def popup_resolution_setter(window: Window):
    """Shows a popup to change the image resolution."""
    experiment = window.get_experiment()
    try:
        image_resolution = experiment.images.resolution()
    except UserError:
        image_resolution = None
    popup = _ResolutionEditorWindow(image_resolution)
    result = popup.exec_()
    if result != QDialog.Accepted:
        return
    new_image_resolution = ImageResolution(popup.x_res.value(), popup.y_res.value(), popup.z_res.value(), popup.t_res.value())
    window.perform_data_action(_UpdateImageResolutionAction(image_resolution, new_image_resolution))
