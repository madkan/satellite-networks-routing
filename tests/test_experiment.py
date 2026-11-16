"""Experiment checks. Run: PYTHONPATH=src python -m unittest discover -s tests -v"""

import unittest
from unittest.mock import patch

from satellite_routing.experiment import generate_packets, run_experiment
from satellite_routing.models import Link, Node


ROUTE = ("G1", "S1", "S2", "G2")


def single_link_network(capacity_bps=8, buffer_capacity_packets=1):
    """Use a tiny network with analytically calculable delivery times."""
    nodes = {
        "G1": Node("G1", "gateway"),
        "G2": Node("G2", "gateway"),
    }
    links = {
        ("G1", "G2"): Link("G1", "G2", capacity_bps, 0.0, buffer_capacity_packets),
    }
    return nodes, links


class ExperimentTests(unittest.TestCase):
    def test_default_case_packet_counts_and_cutoffs(self):
        for interval, count, last_time in [(0.020, 500, 9.980), (0.005, 2000, 9.995)]:
            with self.subTest(interval=interval):
                packets = generate_packets(ROUTE, interval, 10.0)
                self.assertEqual(len(packets), count)
                self.assertEqual([p.packet_id for p in packets], list(range(1, count + 1)))
                self.assertEqual(packets[0].created_at_s, 0.0)
                self.assertAlmostEqual(packets[-1].created_at_s, last_time)
                self.assertTrue(all(p.created_at_s < 10.0 for p in packets))
                self.assertTrue(all(p.size_bits == 8000 and p.route == ROUTE for p in packets))
                for previous, current in zip(packets, packets[1:]):
                    self.assertAlmostEqual(current.created_at_s - previous.created_at_s, interval)

    def test_nonmultiple_duration_and_short_window(self):
        packets = generate_packets(ROUTE, 0.25, 0.6)
        self.assertEqual([p.created_at_s for p in packets], [0.0, 0.25, 0.5])
        self.assertEqual(len(generate_packets(ROUTE, 1.0, 0.1)), 1)

    def test_invalid_generation_settings(self):
        for invalid in [0, -1, float("inf"), float("nan")]:
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    generate_packets(ROUTE, invalid, 1.0)
                with self.assertRaises(ValueError):
                    generate_packets(ROUTE, 0.020, invalid)
        for invalid_size in [0, -8, 8.5, True]:
            with self.subTest(size=invalid_size):
                with self.assertRaises(ValueError):
                    generate_packets(ROUTE, 0.020, 1.0, invalid_size)

    def test_low_load_has_expected_delay_and_throughput(self):
        # Five packets at 0, 20, 40, 60, 80 ms arrive at 18, 38, 58, 78, 98 ms.
        result = run_experiment("small low load", 0.020, duration_s=0.100)
        self.assertEqual(result.route, ROUTE)
        self.assertEqual(result.generated_count, 5)
        self.assertEqual(result.delivered_count, 5)
        self.assertEqual(result.dropped_count, 0)
        self.assertEqual(result.delivered_in_window, 5)
        self.assertAlmostEqual(result.mean_delay_s, 0.018)
        self.assertAlmostEqual(result.mean_queue_wait_s, 0.0)
        self.assertAlmostEqual(result.throughput_bps, 400_000)
        self.assertEqual(result.loss_ratio, 0.0)
        self.assertEqual(result.drain_time_s, 0.0)
        self.assertTrue(result.accounting_passed)

    def test_loss_and_averages_exclude_dropped_packets(self):
        # Packets at 0, .25, .5, .75 s. Each transmission takes 1 s.
        # 1 transmits, 2 waits, 3 and 4 drop. Deliveries occur at 1 and 2 s.
        with patch(
            "satellite_routing.experiment.build_demo_topology",
            return_value=single_link_network(),
        ):
            result = run_experiment("overflow", 0.25, duration_s=1.0, packet_size_bits=8)
        self.assertEqual(result.generated_count, 4)
        self.assertEqual(result.delivered_count, 2)
        self.assertEqual(result.dropped_count, 2)
        self.assertEqual(result.loss_ratio, 0.5)
        self.assertEqual(result.mean_delay_s, (1.0 + 1.75) / 2)
        self.assertEqual(result.mean_queue_wait_s, (0.0 + 0.75) / 2)
        # Delivery exactly at the cutoff is excluded from [0, 1 s).
        self.assertEqual(result.delivered_in_window, 0)
        self.assertEqual(result.throughput_bps, 0.0)
        self.assertEqual(result.drain_time_s, 1.0)
        self.assertTrue(result.accounting_passed)

    def test_throughput_excludes_drain_deliveries(self):
        # Generate at 0, 1, 2 s; deliver at 1, 2, 3 s. Only two arrive before 2.5 s.
        with patch(
            "satellite_routing.experiment.build_demo_topology",
            return_value=single_link_network(),
        ):
            result = run_experiment("drain", 1.0, duration_s=2.5, packet_size_bits=8)
        self.assertEqual(result.delivered_count, 3)
        self.assertEqual(result.delivered_in_window, 2)
        self.assertEqual(result.throughput_bps, 16 / 2.5)
        self.assertEqual(result.offered_load_bps, 8.0)
        self.assertEqual(result.drain_time_s, 0.5)
        self.assertEqual(result.mean_delay_s, 1.0)
        self.assertTrue(result.accounting_passed)

    def test_experiments_start_fresh_after_overload(self):
        before = run_experiment("low", 0.020, duration_s=0.100)
        overloaded = run_experiment("overload", 0.005, duration_s=1.0)
        after = run_experiment("low", 0.020, duration_s=0.100)
        self.assertGreater(overloaded.dropped_count, 0)
        self.assertGreater(overloaded.mean_delay_s, before.mean_delay_s)
        self.assertTrue(overloaded.accounting_passed)
        self.assertEqual(overloaded.generated_count,
                         overloaded.delivered_count + overloaded.dropped_count)
        self.assertEqual(before, after)

    def test_unreachable_destination(self):
        nodes, _ = single_link_network()
        with patch(
            "satellite_routing.experiment.build_demo_topology",
            return_value=(nodes, {}),
        ):
            with self.assertRaises(ValueError):
                run_experiment("unreachable", 0.020)

    def test_low_load_queue_history_covers_full_measurement_window(self):
        result = run_experiment("low", 0.020, duration_s=0.100)
        self.assertEqual(len(result.queue_history), 6)
        self.assertEqual(set(result.queue_history), set(result.buffer_capacities_packets))
        for edge, history in result.queue_history.items():
            self.assertIsInstance(history, tuple)
            self.assertEqual(history[0], (0.0, 0))
            self.assertEqual(history[-1], (0.100, 0))
            self.assertTrue(all(waiting == 0 for _, waiting in history))
            self.assertEqual(result.buffer_capacities_packets[edge], 50)

    def test_overload_history_reaches_capacity_then_drains(self):
        result = run_experiment("overload", 0.005, duration_s=1.0)
        bottleneck = ("S1", "S2")
        history = result.queue_history[bottleneck]
        self.assertEqual(max(q for _, q in history), 50)
        self.assertGreater(result.dropped_count, 0)
        self.assertTrue(any(t > 1.0 and q > 0 for t, q in history))
        self.assertAlmostEqual(history[-1][0], 1.0 + result.drain_time_s)
        self.assertEqual(history[-1][1], 0)
        for edge, samples in result.queue_history.items():
            self.assertTrue(all(0 <= q <= result.buffer_capacities_packets[edge] for _, q in samples))
        # Final records remain empty rather than changing when another run starts.
        run_experiment("next run", 0.020, duration_s=0.100)
        self.assertEqual(result.queue_history[bottleneck], history)


if __name__ == "__main__":
    unittest.main()
