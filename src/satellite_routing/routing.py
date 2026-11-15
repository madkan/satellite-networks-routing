"""Shortest-path routing using propagation delay as the only link cost.

Requires NetworkX. Queue occupancy and transmission capacity do not influence
this baseline's route choice.
"""

import networkx as nx

from .models import Link, Node


def find_shortest_path(
    nodes: dict[str, Node],
    links: dict[tuple[str, str], Link],
    source: str,
    destination: str,
) -> tuple[str, ...] | None:
    """Return the path with minimum total propagation delay, or None if absent.

    All supplied links are treated as available, directed connections. The
    graph is rebuilt on each call to reflect the supplied topology. This
    function does not change any node, link, queue, or packet.

    Unknown endpoints, identical source and destination, and inconsistent
    topology dictionaries raise ValueError. Identical endpoints are rejected
    because Packet routes require at least two nodes. An unreachable, known
    destination instead returns None, allowing the simulator to handle it.
    """
    if source not in nodes or destination not in nodes:
        raise ValueError("source and destination must both exist in nodes")
    if source == destination:
        raise ValueError("source and destination must be different nodes")

    graph = nx.DiGraph()
    for node_id, node in nodes.items():
        if node_id != node.node_id:
            raise ValueError("node dictionary key must match node_id")
        graph.add_node(node_id)

    for endpoints, link in links.items():
        if endpoints != (link.source, link.destination):
            raise ValueError("link dictionary key must match its endpoints")
        if link.source not in nodes or link.destination not in nodes:
            raise ValueError("each link endpoint must exist in nodes")
        graph.add_edge(
            link.source,
            link.destination,
            propagation_delay_s=link.propagation_delay_s,
        )

    try:
        path = nx.dijkstra_path(
            graph, source, destination, weight="propagation_delay_s"
        )
    except nx.NetworkXNoPath:
        return None

    return tuple(path)
