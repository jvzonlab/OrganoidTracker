"""
Package containing the graphical user interface of OrganoidTracker. The menus, the windows, etc.

Note: the Matplotlib-screens that you see are handled by the visualizer package.

To show a popup dialog from within OrganoidTracker:

>>> from organoid_tracker.gui import dialog
>>> dialog.popup_message("My title", "My message")

To start the OrganoidTracker window:

>>> from organoid_tracker.core.experiment import Experiment
>>> experiment = Experiment()  # This empty experiment is a placeholder
>>> from organoid_tracker.gui import main_window
>>> main_window.launch_window(experiment)
"""

APP_NAME = "OrganoidTracker"
