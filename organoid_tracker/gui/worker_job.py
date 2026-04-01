"""A worker job is a job that is applied once to each experiment that is currently open in the GUI. It allows you to
collect some measurement, or make some change to each experiment. It is run on a worker thread, which avoids freezing
the GUI while images are being loaded. The job is applied to each experiment, and the results are passed back to the
GUI.

To get started, make a subclass of WorkerJob, and then call the submit_job function with an instance of your class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union, Sized

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window


class WorkerJob(ABC):
    """A task that makes changes to the experiment. Will be applied to all open tabs."""

    _current_experiment_completed_fraction: Optional[float] = None

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

    def reporting_progress(self, time_points: Union[Iterable[TimePoint], Sized]) -> Iterable[TimePoint]:
        """Updates the progress field as we iterate over the experiment during gather_data. Can be used like
        `for time_point in self.reporting_progress(experiment.time_points()): ...` Only call this once for each
        experiment that you process in gather_data."""
        time_point_count = len(time_points)
        for i, time_point in enumerate(time_points):
            self._current_experiment_completed_fraction = i / time_point_count
            yield time_point

    @abstractmethod
    def on_finished(self, data: Iterable[Any]):
        """Called when the task is finished for all experiments. use_data will already have been called once for every
        experiment."""
        raise NotImplementedError()

    def on_error(self, e: BaseException):
        """Called when an error occurs."""
        from organoid_tracker.gui import dialog
        dialog.popup_exception(e)

    @property
    def current_experiment_completed_fraction(self) -> Optional[float]:
        """Returns a number between 0 and 1, indicating how much of the current experiment has been completed.
        Updated by self.reporting_progress(...). None if the iterator returned by that method has never been started."""
        return self._current_experiment_completed_fraction


def submit_job(window: Window, job: WorkerJob):
    """Submit a job to the worker thread."""
    window.get_scheduler().add_task(_WorkerJobTask(window, job))


class _WorkerJobTask(Task):
    _job: WorkerJob
    _window: Window

    _results_by_tab: Dict[SingleGuiTab, Any]
    _experiment_copies: Dict[SingleGuiTab, Experiment]

    _experiments_done: int = 0

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

            # Increment progress
            self._experiments_done += 1
            self._job._current_experiment_completed_fraction = None

            experiment_copy.close()
        self._experiment_copies.clear()
        return 1

    def get_percentage_completed(self) -> Optional[int]:
        if self._experiments_done == 0 and self._job.current_experiment_completed_fraction is None:
            return None  # Didn't start at all

        # The basis is just how many experiments we already completed
        basis_fraction = self._experiments_done / len(self._experiment_copies)

        # Then we add to that how far along the current experiment we are
        current_experiment_fraction = self._job.current_experiment_completed_fraction
        if current_experiment_fraction is None:
            current_experiment_fraction = 0
        total_fraction = basis_fraction + current_experiment_fraction / len(self._experiment_copies)

        return round(total_fraction * 100)

    def on_finished(self, result: Any):
        for tab in self._results_by_tab:
            self._job.use_data(tab, self._results_by_tab[tab])
        self._job.on_finished(self._results_by_tab.values())

    def on_error(self, e: BaseException):
        self._job.on_error(e)
