"""A worker job is a job that is applied once to each experiment that is currently open in the GUI. It allows you to
collect some measurement, or make some change to each experiment. It is run on a worker thread, which avoids freezing
the GUI while images are being loaded. The job is applied to each experiment, and the results are passed back to the
GUI.

To get started, make a subclass of WorkerJob, and then call the submit_job function with an instance of your class.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Iterable

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window


class WorkerJob(ABC):
    """A task that makes changes to the experiment. Will be applied to all open tabs."""

    @abstractmethod
    def copy_experiment(self, experiment: Experiment) -> Experiment:
        """As gather_data is called on a worker thread, this method is called on the GUI thread to make a copy of the
        experiment. This copy will be passed to gather_data.

        You would normally write something like `return experiment.copy_selected(positions=True)`, to copy whatever data
        you will need on the worker thread. You should copy as few things as possible, as copying may be slow.
        """
        raise NotImplementedError()

    @abstractmethod
    def gather_data(self, experiment_copy: Experiment) -> Any:
        """Gather data from the experiment. Will be called on the worker thread.

        Note: the experiment will be closed (Experiment.close()) after this method is called. This is to avoid memory
        leaks. The original experiment (of the GUI thread) will stay open."""
        raise NotImplementedError()

    def use_data(self, tab: SingleGuiTab, data: Any):
        """Uses the data. Called on the GUI thread, once for every experiment."""
        pass

    @abstractmethod
    def on_finished(self, data: Iterable[Any]):
        """Called when the task is finished for all experiments. use_data will already have been called once for every
        experiment."""
        raise NotImplementedError()

    def on_error(self, e: BaseException):
        """Called when an error occurs."""
        from organoid_tracker.gui import dialog
        dialog.popup_exception(e)


def submit_job(window: Window, job: WorkerJob):
    """Submit a job to the worker thread."""
    window.get_scheduler().add_task(_WorkerJobTask(window, job))


class _WorkerJobTask(Task):
    _job: WorkerJob
    _window: Window

    _results_by_tab: Dict[SingleGuiTab, Any]
    _experiment_copies: Dict[SingleGuiTab, Experiment]

    def __init__(self, window: Window, job: WorkerJob):
        self._job = job
        self._window = window
        self._results_by_tab = dict()
        self._experiment_copies = dict()

        for tab in window.get_gui_experiment().get_active_tabs():
            self._experiment_copies[tab] = job.copy_experiment(tab.experiment)

    def compute(self):
        for tab, experiment_copy in self._experiment_copies.items():
            self._results_by_tab[tab] = self._job.gather_data(experiment_copy)
            experiment_copy.close()
        self._experiment_copies.clear()
        return 1

    def on_finished(self, result: Any):
        for tab in self._results_by_tab:
            self._job.use_data(tab, self._results_by_tab[tab])
        self._job.on_finished(self._results_by_tab.values())

    def on_error(self, e: BaseException):
        self._job.on_error(e)
