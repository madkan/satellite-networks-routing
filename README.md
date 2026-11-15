# Predictive Routing for Dynamic LEO Satellite Networks

An ongoing ECE627 course project investigating whether congestion
forecasting can improve routing in dynamic LEO satellite networks.

## Current implementation

The current stage is a Python discrete-event simulator with:

- Two gateways and four satellites forming two directed alternative paths.
- Shortest-path routing based on total propagation delay.
- Finite FIFO queues associated with outgoing links.
- Separate transmission and propagation delays.
- Multiple packets, queue waiting, and buffer-overflow drops.
- Packet accounting checks after every event.

Satellite motion and machine learning are planned extensions.

## Setup

Tested with Python 3.14.3 and NetworkX 3.7.
Run these commands from the repository root on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install networkx==3.7
```

For subsequent terminal sessions, activate the existing environment:

```bash
source .venv/bin/activate
```

## Run the demo

```bash
PYTHONPATH=src python -m satellite_routing.simulator
```

The demo generates three 8,000-bit packets at 0, 5, and 10 ms.
They follow G1 → S1 → S2 → G2.

| Packet | Creation time | Delivery time | Queue wait | End-to-end delay |
|---|---:|---:|---:|---:|
| 1 | 0 ms | 18 ms | 0 ms | 18 ms |
| 2 | 5 ms | 26 ms | 3 ms | 21 ms |
| 3 | 10 ms | 34 ms | 6 ms | 24 ms |

Expected outcome: three delivered packets, zero drops, and an empty
network after completion.

## Run the tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The 10 tests cover queueing, overflow, concurrent paths, event ordering,
packet sizes, input validation, and single-packet behavior.

## Code organization

- `models.py`: node, packet, and directed-link objects.
- `topology.py`: the fixed demonstration network.
- `routing.py`: propagation-delay shortest-path routing.
- `simulator.py`: event scheduling, packet forwarding, and results.
- `tests/test_simulator.py`: simulator correctness checks.

Source modules are under `src/satellite_routing/`.

## Modeling assumptions

- Topology and link capacities remain fixed during each run.
- The demo uses equal-size packets; packet objects support different sizes.
- Buffer limits count waiting packets, excluding the transmitting packet.
- Each link has an independent transmitter.
- Satellites receive a complete packet before forwarding it.
- Routing uses propagation delay only; measured end-to-end delay also
  includes transmission and queueing.
- All pending events are processed before the run ends.

## Planned work

- Sustained low-load and overloaded traffic experiments.
- Queue-occupancy plots and performance measurements.
- Larger networks with movement-driven connectivity changes.
- Reactive routing comparisons.
- GNN–GRU congestion forecasting and predictive routing evaluation.