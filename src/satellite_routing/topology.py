"""Build the small, fixed network used to learn and verify packet simulation."""

from .models import Link, Node


def build_demo_topology() -> tuple[dict[str, Node], dict[tuple[str, str], Link]]:
    """Return fresh node and directed-link dictionaries for two G1-to-G2 paths.

    G1 -> S1 -> S2 -> G2: 6 ms total propagation delay.
    G1 -> S3 -> S4 -> G2: 12 ms total propagation delay.

    Both paths have a 1 Mbps middle link fed by a 4 Mbps gateway link, so
    overload can build a queue at S1 or S3. Each outgoing queue holds up to
    50 waiting packets, excluding the packet being transmitted.

    These are illustrative settings, not measurements of a constellation.
    Links are one-way; no reverse links are created. Every call creates new
    objects with empty queues and idle transmitters for an independent run.
    """
    nodes = {
        "G1": Node(node_id="G1", kind="gateway"),
        "G2": Node(node_id="G2", kind="gateway"),
        "S1": Node(node_id="S1", kind="satellite"),
        "S2": Node(node_id="S2", kind="satellite"),
        "S3": Node(node_id="S3", kind="satellite"),
        "S4": Node(node_id="S4", kind="satellite"),
    }

    # Each row is (source, destination, capacity in bits/s, delay in seconds).
    link_specs = [
        ("G1", "S1", 4_000_000, 0.002),
        ("S1", "S2", 1_000_000, 0.002),
        ("S2", "G2", 4_000_000, 0.002),
        ("G1", "S3", 4_000_000, 0.004),
        ("S3", "S4", 1_000_000, 0.004),
        ("S4", "G2", 4_000_000, 0.004),
    ]

    links = {}
    for source, destination, capacity_bps, propagation_delay_s in link_specs:
        links[(source, destination)] = Link(
            source=source,
            destination=destination,
            capacity_bps=capacity_bps,
            propagation_delay_s=propagation_delay_s,
            buffer_capacity_packets=50,
        )

    return nodes, links
