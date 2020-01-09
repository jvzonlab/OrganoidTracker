"""Used to find groups of positions that are connected to each other using connections."""
from typing import Optional, Set, List

from ai_track.core import TimePoint
from ai_track.core.connections import Connections
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection


class Cluster:
    """Represents a cluster of positions."""
    positions: Set[Position]

    def __init__(self, *positions: Position):
        self.positions = set(positions)

    def __repr__(self) -> str:
        return "Cluster(" + ", ".join(repr(position) for position in self.positions) + ")"


def _try_remove(set: Set[Position], position: Position):
    """Tries to remove a position from a set, but does nothing if the position is not in the set."""
    if position in set:
        set.remove(position)


def find_clusters(positions: PositionCollection, connections: Connections, time_point: TimePoint) -> List[Cluster]:
    """Returns all clusters - positions that are connected via one or more connections. If position X and position Y are
    in the same cluster, it means that if you follow one or multiple connections, you can get from X to Y."""
    clusters = list()

    positions_without_connections = set(positions.of_time_point(time_point))
    for position_a, position_b in connections.of_time_point(time_point):
        _try_remove(positions_without_connections, position_a)
        _try_remove(positions_without_connections, position_b)

        cluster_of_a: Optional[Cluster] = None
        cluster_of_b: Optional[Cluster] = None
        for cluster in clusters:
            if position_a in cluster.positions:
                cluster_of_a = cluster
            if position_b in cluster.positions:
                cluster_of_b = cluster

        # Now there are four possibilities
        if cluster_of_a is None and cluster_of_b is None:
            # Both are not in a cluster - add new cluster
            clusters.append(Cluster(position_a, position_b))
            continue
        if cluster_of_a is not None and cluster_of_b is None:
            # A is in a cluster, let B join that cluster
            cluster_of_a.positions.add(position_b)
            continue
        if cluster_of_a is None and cluster_of_b is not None:
            # B is in a cluster, let A join that cluster
            cluster_of_b.positions.add(position_a)
            continue
        if cluster_of_a is not None and cluster_of_b is not None:
            # Both are in clusters
            if cluster_of_a is cluster_of_b:
                # It's the same cluster, do nothing
                continue
            else:
                # Merge both clusters
                cluster_of_a.positions |= cluster_of_b.positions
                clusters.remove(cluster_of_b)
                continue
        raise RuntimeError("Should not be able to get here")

    # Add "clusters" of one position for all positions without connections
    for position in positions_without_connections:
        clusters.append(Cluster(position))

    return clusters
