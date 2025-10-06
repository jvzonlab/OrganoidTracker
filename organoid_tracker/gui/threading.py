import queue
from abc import ABC
from queue import Queue
from threading import Thread
from typing import Optional, Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from organoid_tracker.core import UserError
from organoid_tracker.core.concurrent import ConcurrentSet
from organoid_tracker.gui.progress_bar import ProgressBar


class Task(ABC):
    """A long-running task. run() will be called on a worker thread, on_finished() and on_error() on the GUI thread.

    Note: this class is kind of low-level. If you have a task that needs to process Experiment objects in some way,
    consider using the higher-level WorkerJob class instead. This class greatly simplifies running any task on all
    active experiments.
    """

    def compute(self) -> Any:
        raise NotImplementedError()

    def on_finished(self, result: Any):
        raise NotImplementedError()

    def on_error(self, e: BaseException):
        from organoid_tracker.gui import dialog
        dialog.popup_exception(e)

    def get_percentage_completed(self) -> Optional[int]:
        """Gets the percentage currently completed. Can be called from any thread.

        You don't have to implement this method - returning None is fine. However, if you return a value, make sure it's
        100 at the end of the task (so when self.compute() is done).
        """
        return None


class _CompletedTask:
    task: Task
    result: Optional[Any]
    error: Optional[Exception] = None

    def __init__(self, task: Task, result: Optional[Any] = None, error: Optional[Exception] = None):
        if error is None and result is None:
            raise ValueError("Error and result are both None")
        if error is not None and result is not None:
            raise ValueError("Error and result both have a value")
        self.task = task
        self.result = result
        self.error = error

    def handle(self):
        if self.error is not None:
            self.task.on_error(self.error)
        else:
            self.task.on_finished(self.result)


class Scheduler(Thread):
    """To avoid blocking the UI, computationally intensive tasks are run on a worker thread. Simply call add_task(..)
    and the task will be executed on a worker thread."""

    _task_queue: Queue  # Queue[Task]
    _running_task: Optional[Task]
    _finished_queue: Queue  # Queue[_CompletedTask]
    _progress_bar: ProgressBar

    def __init__(self, progress_bar: ProgressBar):
        super().__init__()
        self._task_queue = Queue(maxsize=1)
        self._finished_queue = Queue()
        self._running_task = None
        self._progress_bar = progress_bar

        timer = QTimer(QApplication.instance())
        timer.timeout.connect(self._check_for_results_on_gui_thread)
        timer.start(100)

    def add_task(self, task: Task):
        if self._running_task is None:
            try:
                self._task_queue.put_nowait(task)
                self._progress_bar.set_busy()
                return
            except queue.Full:
                pass
        task.on_error(UserError("Another task is running", "Another task is already running. "
                                                           "Please wait for it to finish."))

    def _check_for_results_on_gui_thread(self):
        try:
            # Update progress bar
            task = self._running_task
            if task is not None:
                self._progress_bar.set_progress(task.get_percentage_completed())

            # Handle all finished tasks
            while True:
                result: _CompletedTask = self._finished_queue.get(block=False)
                self._progress_bar.set_progress(100)
                result.handle()
        except queue.Empty:
            # Ignore, will check again after a while
            pass
        except BaseException as e:
            # Unhandled exception, don't let PyQt catch this
            from organoid_tracker.gui import dialog
            dialog.popup_exception(e)
            self._progress_bar.set_error()

    def _get_percentage_completed(self) -> Optional[float]:
        """Gets the percentage completed of the current task. Returns None if there is no running task, or if the
        current task doesn't keep track of how far it is completed."""
        task = self._running_task
        if task is None:
            return None
        return task.get_percentage_completed()

    def run(self):
        """Long-running method that processes pending tasks. Do not call, let Python call it."""
        while True:
            task: Task = self._task_queue.get(block=True, timeout=None)  # Blocks until the next task
            self._running_task = task  # Set the task as the currently active task
            try:
                result = task.compute()
                self._finished_queue.put(_CompletedTask(task, result=result))
            except Exception as e:
                self._finished_queue.put(_CompletedTask(task, error=e))
            finally:
                self._running_task = None

    def has_active_tasks(self) -> bool:
        """Gets whether there are currently tasks being run or scheduled to run."""
        return self._running_task is not None or self._task_queue.qsize() > 0
