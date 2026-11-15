"""An event-driven simulator for a small batch of packets on fixed routes.

Run the demonstration with:
    PYTHONPATH=src python3 -m satellite_routing.simulator

The demonstration uses NetworkX through routing.py. Packets and their creation
times are supplied explicitly; sustained traffic experiments come later.
"""

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import Literal

from .models import Link, Packet


EventKind = Literal["generate", "transmission_complete", "arrival"]


@dataclass(order=True)
class Event:
    """An action scheduled in simulated seconds.

    Events sort by time, then priority, then insertion sequence. At exactly
    equal timestamps, transmission completions precede arrivals, which precede
    generation. This frees transmitters and queue slots before new admissions.
    Packets and links are excluded from comparisons: they are not sortable.
    """

    time_s: float
    _priority: int = field(init=False, repr=False)
    sequence: int
    kind: EventKind = field(compare=False)
    packet: Packet = field(compare=False)
    link: Link | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self._priority = {"transmission_complete": 0, "arrival": 1, "generate": 2}[self.kind]


class Simulator:
    """Move a batch of packets along assigned routes and record a readable log.

    The supplied links are used directly, so their queues and transmitters
    change during the run. Start with an empty network and fresh packets with
    unique IDs. Each simulator instance runs once and drains all pending events.
    """

    def __init__(self, links: dict[tuple[str, str], Link]) -> None:
        self.links = links
        self.current_time_s = 0.0
        self.delivered_at_s: dict[int, float] = {}
        self.packet_delays_s: dict[int, float] = {}
        self.queue_wait_s: dict[int, float] = {}
        self.dropped_packet_ids: set[int] = set()
        self.event_log: list[str] = []
        self.generated_count = 0
        self.delivered_count = 0
        self.dropped_count = 0
        self._events: list[Event] = []
        self._sequence = count()
        self._in_transit: set[int] = set()
        self._queued_at_s: dict[int, float] = {}
        self._has_run = False

    def run(self, packets: list[Packet]) -> dict[int, float]:
        """Drain the batch; return {packet_id: delay_seconds} for deliveries.

        Dropped packets are excluded from delays and recorded separately in
        dropped_packet_ids. Future scheduled packets count as generated only
        when their generation event occurs. An empty batch returns {}.
        """
        if self._has_run:
            raise ValueError("create a new Simulator for each run")
        if any(
            link.queue_length or link.transmitting_packet is not None
            for link in self.links.values()
        ):
            raise ValueError("simulation must start with an empty network")
        for endpoints, link in self.links.items():
            if endpoints != (link.source, link.destination):
                raise ValueError("link dictionary key must match its endpoints")
        packet_ids = set()
        for packet in packets:
            if packet.packet_id in packet_ids:
                raise ValueError("each packet must have a unique packet_id")
            packet_ids.add(packet.packet_id)
            if packet.current_hop != 0:
                raise ValueError("each packet must start at the first node of its route")
            for edge in zip(packet.route, packet.route[1:]):
                if edge not in self.links:
                    raise ValueError(f"packet route uses a missing link: {edge}")

        self._has_run = True
        for packet in packets:
            self._schedule(packet.created_at_s, "generate", packet)
        while self._events:
            event = heappop(self._events)
            self.current_time_s = event.time_s
            packet = event.packet

            if event.kind == "generate":
                self.generated_count += 1
                self.queue_wait_s[packet.packet_id] = 0.0
                self._log(f"Packet {packet.packet_id} generated at {packet.current_node}.")
                self._enqueue_and_start(packet)
            elif event.kind == "transmission_complete":
                if event.link is None:
                    raise RuntimeError("transmission completion requires a link")
                self._finish_transmission(event.link, packet)
            elif event.kind == "arrival":
                self._arrive(packet)

            self._check_accounting()

        if self.generated_count != self.delivered_count + self.dropped_count:
            raise RuntimeError("events ended with packets still outstanding")
        if self._in_transit or any(
            link.queue_length or link.transmitting_packet is not None
            for link in self.links.values()
        ):
            raise RuntimeError("network is not empty after the run")
        return dict(self.packet_delays_s)

    def _schedule(
        self, time_s: float, kind: EventKind, packet: Packet, link: Link | None = None
    ) -> None:
        """Insert an event; the heap always exposes the earliest event first."""
        if time_s < self.current_time_s:
            raise ValueError("cannot schedule an event in the past")
        heappush(self._events, Event(time_s, next(self._sequence), kind, packet, link))

    def _enqueue_and_start(self, packet: Packet) -> None:
        """Admit to the next queue, or drop on overflow; start only if idle."""
        next_node = packet.next_node
        if next_node is None:
            raise RuntimeError("a delivered packet cannot enter another queue")
        link = self.links[(packet.current_node, next_node)]
        if not link.enqueue(packet):
            self.dropped_count += 1
            self.dropped_packet_ids.add(packet.packet_id)
            self._log(
                f"Packet {packet.packet_id} dropped: queue full for "
                f"{link.source} -> {link.destination}."
            )
            return
        self._queued_at_s[packet.packet_id] = self.current_time_s
        self._log(f"Packet {packet.packet_id} queued for {link.source} -> {link.destination}.")
        self._start_next(link)

    def _start_next(self, link: Link) -> None:
        """Start the oldest waiting packet if the link's transmitter is idle."""
        packet = link.start_transmission()
        if packet is None:
            return
        waited_s = self.current_time_s - self._queued_at_s.pop(packet.packet_id)
        self.queue_wait_s[packet.packet_id] += waited_s
        self._log(
            f"Packet {packet.packet_id} transmission starts: "
            f"{link.source} -> {link.destination}; waited {waited_s * 1000:.3f} ms."
        )
        self._schedule(
            self.current_time_s + link.transmission_delay_s(packet),
            "transmission_complete",
            packet,
            link,
        )

    def _finish_transmission(self, link: Link, packet: Packet) -> None:
        """Release the transmitter, then schedule arrival after propagation."""
        if link.transmitting_packet is not packet:
            raise RuntimeError("unexpected packet in transmitter")
        link.finish_transmission()
        self._in_transit.add(packet.packet_id)
        self._log(
            f"Packet {packet.packet_id} transmission finishes: "
            f"{link.source} -> {link.destination}; "
            "packet begins propagating."
        )
        self._schedule(self.current_time_s + link.propagation_delay_s, "arrival", packet)
        # The previous packet propagates independently; the link can send again.
        self._start_next(link)

    def _arrive(self, packet: Packet) -> None:
        """Advance a hop only after propagation; deliver or start the next hop."""
        self._in_transit.remove(packet.packet_id)
        packet.advance_hop()
        self._log(f"Packet {packet.packet_id} arrives at {packet.current_node}.")
        if packet.next_node is None:
            self.delivered_count += 1
            self.delivered_at_s[packet.packet_id] = self.current_time_s
            delay_s = self.current_time_s - packet.created_at_s
            self.packet_delays_s[packet.packet_id] = delay_s
            self._log(
                f"Packet {packet.packet_id} delivered. "
                f"End-to-end delay: {delay_s * 1000:.3f} ms."
            )
        else:
            self._enqueue_and_start(packet)

    def _check_accounting(self) -> None:
        """Check conservation and buffer limits after every completed event."""
        waiting = sum(link.queue_length for link in self.links.values())
        sending = sum(link.transmitting_packet is not None for link in self.links.values())
        accounted = (
            self.delivered_count + self.dropped_count + waiting + sending
            + len(self._in_transit)
        )
        if self.generated_count != accounted:
            raise RuntimeError("packet accounting failed")
        if len(self._queued_at_s) != waiting:
            raise RuntimeError("queue timing records do not match waiting packets")
        if any(link.queue_length > link.buffer_capacity_packets for link in self.links.values()):
            raise RuntimeError("queue exceeds its buffer capacity")

    def _log(self, message: str) -> None:
        self.event_log.append(f"{self.current_time_s * 1000:8.3f} ms: {message}")


def main() -> None:
    """Send three 8,000-bit packets, 5 ms apart, and print their journeys."""
    from .routing import find_shortest_path
    from .topology import build_demo_topology

    nodes, links = build_demo_topology()
    route = find_shortest_path(nodes, links, "G1", "G2")
    if route is None:
        raise RuntimeError("demo topology has no G1-to-G2 path")
    packets = [
        Packet(1, "flow-1", 8000, 0.000, route),
        Packet(2, "flow-1", 8000, 0.005, route),
        Packet(3, "flow-1", 8000, 0.010, route),
    ]
    simulator = Simulator(links)
    delays = simulator.run(packets)
    print("Route:", " -> ".join(route))
    print("\n".join(simulator.event_log))
    print("\nPacket   Created (ms)   Delivered (ms)   Queue wait (ms)   Delay (ms)")
    for packet in packets:
        packet_id = packet.packet_id
        if packet_id in delays:
            print(
                f"{packet_id:6d}   {packet.created_at_s * 1000:12.3f}   "
                f"{simulator.delivered_at_s[packet_id] * 1000:14.3f}   "
                f"{simulator.queue_wait_s[packet_id] * 1000:15.3f}   "
                f"{delays[packet_id] * 1000:10.3f}"
            )
    print(
        f"Generated: {simulator.generated_count}; delivered: {simulator.delivered_count}; "
        f"dropped: {simulator.dropped_count}."
    )
    print("Accounting passed after every event; all queues empty and transmitters idle.")


if __name__ == "__main__":
    main()
