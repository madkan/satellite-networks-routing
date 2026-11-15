"""Run with: PYTHONPATH=src python3 -m unittest discover -s tests -v"""

import unittest

from satellite_routing.models import Link, Packet
from satellite_routing.simulator import Simulator
from satellite_routing.topology import build_demo_topology


UPPER_ROUTE = ("G1", "S1", "S2", "G2")


class SimulatorTests(unittest.TestCase):
    def assert_drained(self, simulator):
        self.assertEqual(
            simulator.generated_count,
            simulator.delivered_count + simulator.dropped_count,
        )
        for link in simulator.links.values():
            self.assertEqual(link.queue_length, 0)
            self.assertIsNone(link.transmitting_packet)

    def test_three_packets_wait_at_bottleneck(self):
        _, links = build_demo_topology()
        packets = [
            Packet(i + 1, "flow-1", 8000, i * 0.005, UPPER_ROUTE)
            for i in range(3)
        ]
        simulator = Simulator(links)
        delays = simulator.run(packets)
        for packet_id, arrival, delay, wait in [
            (1, 0.018, 0.018, 0.000),
            (2, 0.026, 0.021, 0.003),
            (3, 0.034, 0.024, 0.006),
        ]:
            self.assertAlmostEqual(simulator.delivered_at_s[packet_id], arrival)
            self.assertAlmostEqual(delays[packet_id], delay)
            self.assertAlmostEqual(simulator.queue_wait_s[packet_id], wait)
        self.assertEqual(simulator.generated_count, 3)
        self.assertEqual(simulator.dropped_count, 0)
        self.assert_drained(simulator)

    def test_single_packet_still_takes_18_ms(self):
        _, links = build_demo_topology()
        simulator = Simulator(links)
        delays = simulator.run([Packet(1, "f", 8000, 1.0, UPPER_ROUTE)])
        self.assertAlmostEqual(delays[1], 0.018)
        self.assertAlmostEqual(simulator.delivered_at_s[1], 1.018)
        self.assert_drained(simulator)

    def test_fifo_overflow_and_transmission_during_propagation(self):
        # Each transmission lasts 1 s, followed by 10 s of propagation.
        # One waiting slot accepts packet 2 while packet 1 transmits; 3 drops.
        link = Link("A", "B", 8, 10, 1)
        simulator = Simulator({("A", "B"): link})
        delays = simulator.run([
            Packet(i, "f", 8, 0.0, ("A", "B")) for i in (1, 2, 3)
        ])
        self.assertEqual(delays, {1: 11.0, 2: 12.0})
        self.assertEqual(simulator.queue_wait_s[2], 1.0)
        self.assertEqual(simulator.dropped_packet_ids, {3})
        self.assertEqual(simulator.dropped_count, 1)
        self.assert_drained(simulator)

    def test_completion_frees_queue_before_same_time_generation(self):
        link = Link("A", "B", 8, 10, 1)
        simulator = Simulator({("A", "B"): link})
        delays = simulator.run([
            Packet(1, "f", 8, 0.0, ("A", "B")),
            Packet(2, "f", 8, 0.0, ("A", "B")),
            Packet(3, "f", 8, 1.0, ("A", "B")),
        ])
        self.assertEqual(delays, {1: 11.0, 2: 12.0, 3: 12.0})
        self.assertEqual(simulator.dropped_count, 0)
        self.assert_drained(simulator)

    def test_overflow_at_satellite(self):
        # All three fit at A. A->S serves one per second; S->B needs 4 s.
        links = {
            ("A", "S"): Link("A", "S", 8, 0, 3),
            ("S", "B"): Link("S", "B", 2, 0, 1),
        }
        simulator = Simulator(links)
        delays = simulator.run([
            Packet(i, "f", 8, 0.0, ("A", "S", "B")) for i in (1, 2, 3)
        ])
        self.assertEqual(delays, {1: 5.0, 2: 9.0})
        self.assertEqual(simulator.dropped_packet_ids, {3})
        self.assert_drained(simulator)

    def test_independent_paths_can_run_concurrently(self):
        _, links = build_demo_topology()
        simulator = Simulator(links)
        delays = simulator.run([
            Packet(1, "upper", 8000, 0.0, UPPER_ROUTE),
            Packet(2, "lower", 8000, 0.0, ("G1", "S3", "S4", "G2")),
        ])
        self.assertAlmostEqual(delays[1], 0.018)
        self.assertAlmostEqual(delays[2], 0.024)
        self.assertEqual(simulator.queue_wait_s, {1: 0.0, 2: 0.0})
        self.assert_drained(simulator)

    def test_generation_is_ordered_by_time_and_sizes_affect_service(self):
        simulator = Simulator({("A", "B"): Link("A", "B", 8, 0, 2)})
        # Input order is intentionally not chronological. Packet 1 sends 16
        # bits in 2 s, so packet 2, created at 1 s, waits 1 s then sends for 1 s.
        delays = simulator.run([
            Packet(2, "f", 8, 1.0, ("A", "B")),
            Packet(1, "f", 16, 0.0, ("A", "B")),
        ])
        self.assertEqual(delays, {1: 2.0, 2: 2.0})
        self.assertEqual(simulator.queue_wait_s[2], 1.0)
        self.assert_drained(simulator)

    def test_duplicate_ids_rejected_before_network_changes(self):
        _, links = build_demo_topology()
        simulator = Simulator(links)
        with self.assertRaises(ValueError):
            simulator.run([
                Packet(1, "f", 8000, 0.0, UPPER_ROUTE),
                Packet(1, "f", 8000, 0.005, UPPER_ROUTE),
            ])
        self.assertEqual(simulator.generated_count, 0)
        self.assert_drained(simulator)

    def test_missing_link_rejected_before_network_changes(self):
        _, links = build_demo_topology()
        simulator = Simulator(links)
        with self.assertRaises(ValueError):
            simulator.run([Packet(1, "f", 8000, 0.0, ("G1", "G2"))])
        self.assertEqual(simulator.generated_count, 0)
        self.assert_drained(simulator)

    def test_empty_batch_and_reuse(self):
        simulator = Simulator({})
        self.assertEqual(simulator.run([]), {})
        self.assert_drained(simulator)
        with self.assertRaises(ValueError):
            simulator.run([])


if __name__ == "__main__":
    unittest.main()
