from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window):
    return {
        "Edit//Batch-Establish connections between positions//Connect positions by distance...":
            lambda: _connect_positions_by_distance(window),
        "Edit//Batch-Establish connections between positions//Connect positions by distance and number...":
            lambda: _connect_positions_by_distance_and_number(window),
        "Edit//Batch-Establish connections between positions//Connect positions by angle...":
            lambda: _connect_positions_by_angle(window),
    }


class _ReplaceConnectionsAction(UndoableAction):

    _old_connections: Connections
    _new_connections: Connections

    def __init__(self, old_connections: Connections, new_connections: Connections):
        self._old_connections = old_connections
        self._new_connections = new_connections

    def do(self, experiment: Experiment) -> str:
        experiment.connections = self._new_connections
        return f"Created {len(self._new_connections)} new connections"

    def undo(self, experiment: Experiment) -> str:
        experiment.connections = self._old_connections
        return "Restored the previous connections"


def _connect_positions_by_distance(window: Window):
    """Strictly by distance."""
    distance_um = dialog.prompt_float("Maximum distance", "Up to what distance (μm) should all positions be"
                                                          " connected?", minimum=0, default=15)
    if distance_um is None:
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        from organoid_tracker.connecting.connector_by_distance import ConnectorByDistance
        connector = ConnectorByDistance(distance_um)
        connections = connector.create_connections(tab.experiment)
        result_message = tab.undo_redo.do(_ReplaceConnectionsAction(tab.experiment.connections, connections), tab.experiment)
        window.set_status(result_message)


def _connect_positions_by_distance_and_number(window: Window):
    """By number & distance."""
    distance_um = dialog.prompt_float("Maximum distance", "Up to what distance (μm) should all positions be"
                                                          " connected?", minimum=0, default=15)
    if distance_um is None:
        return
    max_number = dialog.prompt_int("Maximum distance", "What is the maximum number of connections that a\n"
                                                       "cell can make? (A cell can still end up receiving\n"
                                                       "more.)", minimum=0, default=5)
    if max_number is None:
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        from organoid_tracker.connecting.connector_by_distance import ConnectorByDistance
        connector = ConnectorByDistance(distance_um, max_number)
        connections = connector.create_connections(tab.experiment)
        result_message = tab.undo_redo.do(_ReplaceConnectionsAction(tab.experiment.connections, connections), tab.experiment)
        window.set_status(result_message)


def _connect_positions_by_angle(window: Window):
    if not dialog.popup_message_cancellable("Automatic neighbor detection", "This algorithm will look at the 10 nearest"
            " cells of any cell. If there are no cells in between that cell and any of the nearest cells, the cells are"
            " considered neighbors. The algorithm is not exact, as it uses only the cell center positions, not the full"
            " shape of the cells.\n\nNote: this computation might take a few minutes."):
        return

    for tab in window.get_gui_experiment().get_active_tabs():
        from organoid_tracker.connecting import connector_using_angles
        connections = connector_using_angles.create_connections(tab.experiment, print_progress=True)
        result_message = tab.undo_redo.do(_ReplaceConnectionsAction(tab.experiment.connections, connections), tab.experiment)
        window.set_status(result_message)


