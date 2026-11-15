"""Network objects for the first, fixed-topology simulator.

Units are bits, bits/second, and seconds. A link's buffer limit counts waiting
packets only; the packet being transmitted occupies a separate slot. These
objects hold state but do not advance time or schedule packet arrivals.
"""

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class Node:
    """A gateway or satellite, identified by a unique name in the network."""

    node_id: str
    kind: Literal["gateway", "satellite"]

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if self.kind not in ("gateway", "satellite"):
            raise ValueError("kind must be 'gateway' or 'satellite'")


@dataclass
class Packet:
    """A packet following a fixed route of node IDs.

    current_hop is the index of the last node reached. It changes on arrival
    at the next node, never when transmission begins or ends.
    """

    packet_id: int
    flow_id: str
    size_bits: int
    created_at_s: float
    route: tuple[str, ...]
    current_hop: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.size_bits, int) or self.size_bits <= 0:
            raise ValueError("size_bits must be a positive integer")
        if not isfinite(self.created_at_s) or self.created_at_s < 0:
            raise ValueError("created_at_s must be finite and nonnegative")
        self.route = tuple(self.route)
        if len(self.route) < 2 or any(not node_id for node_id in self.route):
            raise ValueError("route must contain at least two nonempty node IDs")
        if len(set(self.route)) != len(self.route):
            raise ValueError("route must not repeat a node")

    @property
    def current_node(self) -> str:
        """The last node this packet reached, including while in transit."""
        return self.route[self.current_hop]

    @property
    def next_node(self) -> str | None:
        """The next node on the route, or None at the destination."""
        if self.current_hop == len(self.route) - 1:
            return None
        return self.route[self.current_hop + 1]

    def advance_hop(self) -> None:
        """Record arrival at the next node; the future simulator calls this."""
        if self.next_node is None:
            raise ValueError("packet has already reached its destination")
        self.current_hop += 1


@dataclass
class Link:
    """A directed connection with a finite FIFO queue and one transmitter.

    A -> B and B -> A would be separate Link objects. A busy transmitter
    cannot start another packet until finish_transmission() is called.
    """

    source: str
    destination: str
    capacity_bps: float
    propagation_delay_s: float
    buffer_capacity_packets: int
    _queue: deque[Packet] = field(default_factory=deque, init=False, repr=False)
    _transmitting_packet: Packet | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.source or not self.destination or self.source == self.destination:
            raise ValueError("link endpoints must be nonempty and distinct")
        if not isfinite(self.capacity_bps) or self.capacity_bps <= 0:
            raise ValueError("capacity_bps must be finite and positive")
        if not isfinite(self.propagation_delay_s) or self.propagation_delay_s < 0:
            raise ValueError("propagation_delay_s must be finite and nonnegative")
        if (
            not isinstance(self.buffer_capacity_packets, int)
            or self.buffer_capacity_packets <= 0
        ):
            raise ValueError("buffer_capacity_packets must be a positive integer")

    @property
    def queue_length(self) -> int:
        """Number of waiting packets, excluding the transmitting packet."""
        return len(self._queue)

    @property
    def queue_occupancy_bits(self) -> int:
        """Total bits waiting in this link's queue."""
        return sum(packet.size_bits for packet in self._queue)

    @property
    def transmitting_packet(self) -> Packet | None:
        """Packet currently being serialized onto the link, if any."""
        return self._transmitting_packet

    def enqueue(self, packet: Packet) -> bool:
        """Append a packet to the queue; return False if the buffer is full.

        The simulator will count a full-buffer rejection as a dropped packet.
        A packet routed to a different link is a programming error instead.
        """
        if (packet.current_node, packet.next_node) != (self.source, self.destination):
            raise ValueError("packet's next hop does not match this link")
        if packet is self._transmitting_packet or any(
            queued is packet for queued in self._queue
        ):
            raise ValueError("packet is already on this link")
        if self.queue_length >= self.buffer_capacity_packets:
            return False
        self._queue.append(packet)
        return True

    def start_transmission(self) -> Packet | None:
        """Move the oldest waiting packet into the idle transmitter.

        Return None if the transmitter is busy or the queue is empty.
        The simulator will schedule completion using transmission_delay_s().
        """
        if self._transmitting_packet is not None or not self._queue:
            return None
        self._transmitting_packet = self._queue.popleft()
        return self._transmitting_packet

    def transmission_delay_s(self, packet: Packet) -> float:
        """Time to serialize this packet: packet bits / link bits per second."""
        return packet.size_bits / self.capacity_bps

    def finish_transmission(self) -> Packet:
        """Free the transmitter and return the packet that will propagate.

        This does not advance the packet's hop. Arrival occurs only after the
        propagation delay, which the simulator will schedule separately.
        """
        if self._transmitting_packet is None:
            raise ValueError("there is no transmission to finish")
        packet = self._transmitting_packet
        self._transmitting_packet = None
        return packet
