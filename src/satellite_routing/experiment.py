"""Compare sustained traffic loads using the existing shortest-path simulator.

Run from the repository root:
    PYTHONPATH=src python -m satellite_routing.experiment

Generation and throughput use the interval [0, duration_s). The simulator then
drains all remaining packets. Final counts and delivered-packet averages include
that drain period. Runs start empty, with regular arrivals and no warm-up period.
"""

from dataclasses import dataclass
from math import isfinite
from statistics import mean

from .models import Packet
from .routing import find_shortest_path
from .simulator import Simulator
from .topology import build_demo_topology


@dataclass(frozen=True)
class ExperimentResult:
    """Summary of one completed run; times are seconds and rates are bits/s."""

    name: str
    route: tuple[str, ...]
    duration_s: float
    offered_load_bps: float  # Configured packet size / generation interval.
    generated_count: int
    delivered_count: int  # Includes deliveries during draining.
    dropped_count: int
    delivered_in_window: int  # Delivered strictly before duration_s.
    loss_ratio: float
    mean_delay_s: float | None  # None if no packets were delivered.
    mean_queue_wait_s: float | None  # Delivered packets only, across all hops.
    throughput_bps: float  # Payload delivered in the window / duration_s.
    drain_time_s: float
    accounting_passed: bool


def generate_packets(
    route: tuple[str, ...],
    interval_s: float,
    duration_s: float,
    packet_size_bits: int = 8000,
) -> list[Packet]:
    """Create one flow of equal-size packets at 0, interval_s, 2*interval_s, ...

    Stop strictly before duration_s. IDs start at 1. Multiplying the index by
    the interval avoids the drift from repeatedly adding the interval to time.
    """
    if not isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be finite and positive")
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if isinstance(packet_size_bits, bool) or not isinstance(packet_size_bits, int):
        raise ValueError("packet_size_bits must be a positive integer")
    if packet_size_bits <= 0:
        raise ValueError("packet_size_bits must be a positive integer")

    packets = []
    index = 0
    while index * interval_s < duration_s:
        packets.append(Packet(
            packet_id=index + 1,
            flow_id="flow-1",
            size_bits=packet_size_bits,
            created_at_s=index * interval_s,
            route=route,
        ))
        index += 1
    return packets


def run_experiment(
    name: str,
    interval_s: float,
    duration_s: float = 10.0,
    packet_size_bits: int = 8000,
) -> ExperimentResult:
    """Build a fresh demo network, run the traffic, and summarize measurements.

    The nominal offered load is packet_size_bits / interval_s. For durations
    that are not exact multiples of the interval, the finite-run generated
    bits / duration can differ from this configured rate.
    """
    nodes, links = build_demo_topology()
    route = find_shortest_path(nodes, links, "G1", "G2")
    if route is None:
        raise ValueError("the experiment needs an available G1-to-G2 path")
    packets = generate_packets(route, interval_s, duration_s, packet_size_bits)
    simulator = Simulator(links)
    delays = simulator.run(packets)

    # Keep window throughput separate from final delivery counts after draining.
    delivered_in_window = sum(
        arrival_s < duration_s for arrival_s in simulator.delivered_at_s.values()
    )
    throughput_bps = delivered_in_window * packet_size_bits / duration_s
    mean_delay_s = mean(delays.values()) if delays else None
    mean_queue_wait_s = (
        mean(simulator.queue_wait_s[packet_id] for packet_id in delays)
        if delays else None
    )

    # In addition to the simulator's per-event counts, check that each input
    # packet ID appears in exactly one final outcome: delivered or dropped.
    generated_ids = {packet.packet_id for packet in packets}
    delivered_ids = set(delays)
    dropped_ids = simulator.dropped_packet_ids
    accounting_passed = (
        simulator.generated_count == len(packets)
        and simulator.delivered_count == len(delivered_ids)
        and simulator.dropped_count == len(dropped_ids)
        and delivered_ids.isdisjoint(dropped_ids)
        and (delivered_ids | dropped_ids) == generated_ids
        and all(
            link.queue_length == 0 and link.transmitting_packet is None
            for link in links.values()
        )
    )
    if not accounting_passed:
        raise RuntimeError("final experiment packet accounting failed")

    return ExperimentResult(
        name=name,
        route=route,
        duration_s=duration_s,
        offered_load_bps=packet_size_bits / interval_s,
        generated_count=simulator.generated_count,
        delivered_count=simulator.delivered_count,
        dropped_count=simulator.dropped_count,
        delivered_in_window=delivered_in_window,
        loss_ratio=simulator.dropped_count / simulator.generated_count,
        mean_delay_s=mean_delay_s,
        mean_queue_wait_s=mean_queue_wait_s,
        throughput_bps=throughput_bps,
        # The last event may occur before generation's nominal cutoff at low load.
        drain_time_s=max(0.0, simulator.current_time_s - duration_s),
        accounting_passed=accounting_passed,
    )


def main() -> None:
    """Print a comparison table without printing thousands of packet events."""
    results = [
        run_experiment("Low load", interval_s=0.020),
        run_experiment("Overload", interval_s=0.005),
    ]
    print("Fixed topology; shortest path:", " -> ".join(results[0].route))
    print("8,000-bit packets; 10 s of traffic; 1 Mbps path bottleneck.")
    print("Throughput counts deliveries in [0, 10 s); final metrics include draining.\n")

    # Values stay in seconds and bits/s internally; only the display converts units.
    rows = [
        ("Offered load (Mbps, nominal)", [f"{r.offered_load_bps / 1e6:.3f}" for r in results]),
        ("Generated packets", [str(r.generated_count) for r in results]),
        ("Delivered packets (final)", [str(r.delivered_count) for r in results]),
        ("Dropped packets (final)", [str(r.dropped_count) for r in results]),
        ("Delivered before 10 s", [str(r.delivered_in_window) for r in results]),
        ("Loss (%)", [f"{r.loss_ratio * 100:.3f}" for r in results]),
        ("Mean delay (ms, delivered)", [
            "N/A" if r.mean_delay_s is None else f"{r.mean_delay_s * 1000:.3f}"
            for r in results
        ]),
        ("Mean queue wait (ms, delivered)", [
            "N/A" if r.mean_queue_wait_s is None else f"{r.mean_queue_wait_s * 1000:.3f}"
            for r in results
        ]),
        ("Throughput (Mbps, first 10 s)", [f"{r.throughput_bps / 1e6:.3f}" for r in results]),
        ("Extra drain time (ms)", [f"{r.drain_time_s * 1000:.3f}" for r in results]),
        ("Final accounting", ["PASS" if r.accounting_passed else "FAIL" for r in results]),
    ]
    print(f"{'Metric':<34}{results[0].name:>15}{results[1].name:>15}")
    print("-" * 64)
    for label, values in rows:
        print(f"{label:<34}{values[0]:>15}{values[1]:>15}")


if __name__ == "__main__":
    main()
