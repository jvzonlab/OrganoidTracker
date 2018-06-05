import queue
import tkinter
from queue import Queue
from threading import Thread
from typing import Optional, Any

from core import UserError
from core.concurrent import ConcurrentSet


class Task:
    """A long-running task. run() will be called on a worker thread, on_Finished() and on_error() on the GUI thread."""
    def run(self) -> Any:
        raise NotImplementedError()

    def on_finished(self, result: Any):
        raise NotImplementedError()

    def on_error(self, e: BaseException):
        from gui import dialog
        if isinstance(e, UserError):
            dialog.popup_error(e.title, e.body)
        else:
            dialog.popup_exception(e)


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
    _running_tasks: ConcurrentSet
    _finished_queue: Queue  # Queue[_CompletedTask]
    _tk_root: tkinter.Tk

    def __init__(self, tk_root: tkinter.Tk):
        super().__init__()
        self._tk_root = tk_root
        self._task_queue = Queue(maxsize=1)
        self._finished_queue = Queue()
        self._running_tasks = ConcurrentSet()

        self._check_for_results_on_gui_thread()

    def add_task(self, task: Task):
        if len(self._running_tasks) == 0:
            try:
                self._task_queue.put_nowait(task)
                return
            except queue.Full:
                pass
        task.on_error(UserError("Another task is running", "Another task is already running. "
                                                           "Please wait for it to finish."))

    def _check_for_results_on_gui_thread(self):
        try:
            while True:
                result: _CompletedTask = self._finished_queue.get(block = False)
                result.handle()
        except queue.Empty:
            # Check again after a while
            self._tk_root.after(100, self._check_for_results_on_gui_thread)

    def run(self):
        """Long running method that processes pending tasks. Do not call, let Python call it."""
        while True:
            task: Task = self._task_queue.get(block=True, timeout=None)
            self._running_tasks.add(task)
            try:
                result = task.run()
                self._finished_queue.put(_CompletedTask(task, result=result))
            except Exception as e:
                self._finished_queue.put(_CompletedTask(task, error=e))
            finally:
                self._running_tasks.remove(task)

    def has_active_tasks(self) -> bool:
        """Gets whether there are currently tasks being run or scheduled to run."""
        return len(self._running_tasks) > 0 or self._task_queue.qsize() > 0