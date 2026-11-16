"""Plot queue occupancy for the low-load and overloaded experiments.

Run from the repository root:
    PYTHONPATH=src python -m satellite_routing.plotting

The PNG is saved in results/ and can be committed with the source code.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save images without requiring a desktop plot window.
import matplotlib.pyplot as plt

from .experiment import ExperimentResult, run_experiment


def _queue_series(
    result: ExperimentResult, edge: tuple[str, str], end_time_s: float
) -> tuple[list[float], list[int]]:
    """Use the last state at each timestamp and extend the drained, empty queue.

    Events at one timestamp can have intermediate queue states. Only the final
    state persists until time advances, so it is the value used for the plot.
    """
    times = []
    occupancies = []
    for time_s, waiting in result.queue_history[edge]:
        if times and time_s == times[-1]:
            occupancies[-1] = waiting
        else:
            times.append(time_s)
            occupancies.append(waiting)
    if times[-1] < end_time_s:
        # run_experiment drains the network, so its final waiting count is zero.
        times.append(end_time_s)
        occupancies.append(occupancies[-1])
    return times, occupancies


def plot_queue_comparison(
    low_load: ExperimentResult,
    overload: ExperimentResult,
    output_path: str | Path = "results/queue_occupancy_low_vs_overload.png",
    edge: tuple[str, str] = ("S1", "S2"),
) -> Path:
    """Save a step plot with a full-run view and a zoom around traffic cutoff.

    Both experiments must use the same queue capacity and generation duration.
    Counts include waiting packets only, not packets transmitting or propagating.
    """
    for result in (low_load, overload):
        if edge not in result.queue_history or not result.queue_history[edge]:
            raise ValueError(f"missing queue history for {edge} in {result.name}")
    capacity = low_load.buffer_capacities_packets[edge]
    if capacity != overload.buffer_capacities_packets[edge]:
        raise ValueError("comparison requires matching buffer capacities")
    if low_load.duration_s != overload.duration_s:
        raise ValueError("comparison requires matching traffic durations")

    cutoff_s = low_load.duration_s
    last_event_s = max(
        low_load.queue_history[edge][-1][0],
        overload.queue_history[edge][-1][0],
    )
    end_time_s = max(cutoff_s, last_event_s) + 0.1
    low_times, low_counts = _queue_series(low_load, edge, end_time_s)
    high_times, high_counts = _queue_series(overload, edge, end_time_s)

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5.8), sharey=True,
            gridspec_kw={"width_ratios": [2.2, 1]},
        )
        try:
            fig.suptitle(f"Queue occupancy at {edge[0]} → {edge[1]}", fontsize=19, fontweight="bold", y=0.96)
            fig.text(
                0.5, 0.88,
                "Shortest-path routing · Fixed topology · Waiting packets only",
                ha="center", color="#475569",
            )
            for ax in axes:
                ax.axvspan(cutoff_s, end_time_s, color="#f1f5f9", zorder=0)
                # Post steps keep each recorded value until the next event.
                ax.step(low_times, low_counts, where="post", color="#2563eb", linewidth=2,
                        label=f"Low load ({low_load.offered_load_bps / 1e6:g} Mbps)")
                ax.step(high_times, high_counts, where="post", color="#c2410c", linewidth=1.5,
                        label=f"Overload ({overload.offered_load_bps / 1e6:g} Mbps)")
                ax.axhline(capacity, color="#64748b", linestyle="--", linewidth=1.2,
                           label=f"Buffer limit ({capacity} packets)")
                ax.axvline(cutoff_s, color="#334155", linestyle=":", linewidth=1.5,
                           label=f"Generation stops ({cutoff_s:g} s)")
                ax.set_xlabel("Simulation time (s)")
                ax.set_ylim(-2, capacity + 5)
                ax.set_yticks(range(0, capacity + 1, max(1, capacity // 5)))
                ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
                ax.set_axisbelow(True)

            axes[0].set_title("Full experiment", loc="left", fontsize=12, pad=12)
            axes[0].set_ylabel("Packets waiting in the outgoing queue")
            axes[0].set_xlim(0, end_time_s)
            axes[1].set_title("Traffic cutoff and draining", loc="left", fontsize=12, pad=12)
            axes[1].set_xlim(max(0, cutoff_s - 0.2), end_time_s)
            axes[1].tick_params(axis="x", labelsize=10)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.08),
                       ncol=2, frameon=False, fontsize=10)
            fig.text(
                0.5, 0.025,
                "Shaded region: generation has stopped; packets already in the network continue to drain.",
                ha="center", fontsize=10, color="#475569",
            )
            fig.subplots_adjust(left=0.085, right=0.98, top=0.77, bottom=0.28, wspace=0.12)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, facecolor="white")
        finally:
            plt.close(fig)
    return output_path


def main() -> None:
    """Run both 10-second cases and save the comparison figure."""
    low_load = run_experiment("Low load", interval_s=0.020)
    overload = run_experiment("Overload", interval_s=0.005)
    output_path = plot_queue_comparison(low_load, overload)
    print(f"Saved plot: {output_path.resolve()}")


if __name__ == "__main__":
    main()
