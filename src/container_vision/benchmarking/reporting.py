"""Generate durable benchmark summaries and plots from CSV files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_rows(rows: list[dict], fields: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
        summary[field] = {
            "average": mean(values) if values else 0.0,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "maximum": max(values) if values else 0.0,
        }
    return summary


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _plot_imports():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_run_plots(run_dir: Path) -> list[str]:
    plt = _plot_imports()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames = [row for row in read_csv(run_dir / "frame_metrics.csv") if row.get("warmup") == "0"]
    resources = read_csv(run_dir / "resource_metrics.csv")
    generated = []

    timing_fields = [
        "decode_ms", "queue_wait_ms", "yolo_preprocess_ms", "yolo_inference_ms",
        "yolo_postprocess_ms", "crop_extraction_ms", "ocr_preprocess_ms",
        "ocr_inference_ms", "ocr_postprocess_ms", "rules_event_ms",
        "frame_processing_ms", "end_to_end_ms",
    ]
    if frames:
        timing = summarize_rows(frames, timing_fields)
        labels = [field.removesuffix("_ms").replace("_", "\n") for field in timing_fields]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.bar([value - 0.25 for value in x], [timing[f]["average"] for f in timing_fields], 0.25, label="average")
        ax.bar(x, [timing[f]["p95"] for f in timing_fields], 0.25, label="p95")
        ax.bar([value + 0.25 for value in x], [timing[f]["p99"] for f in timing_fields], 0.25, label="p99")
        ax.set_xticks(list(x), labels)
        ax.set_ylabel("Milliseconds")
        ax.set_title("Pipeline latency by stage")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        target = plots_dir / "stage_latency.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        generated.append(str(target.relative_to(run_dir)))

        times = [float(row["elapsed_s"]) for row in frames]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(times, [float(row["end_to_end_ms"]) for row in frames], linewidth=0.6)
        ax.set(title="End-to-end frame latency", xlabel="Elapsed seconds", ylabel="Milliseconds")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        target = plots_dir / "latency_over_time.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        generated.append(str(target.relative_to(run_dir)))

    if resources:
        times = [float(row["elapsed_s"]) for row in resources]
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(times, [float(row["cpu_percent"] or 0) for row in resources], label="CPU %")
        axes[0].plot(times, [float(row["ram_percent"] or 0) for row in resources], label="RAM %")
        axes[1].plot(times, [float(row["gpu_util_percent"] or 0) for row in resources], label="GPU %")
        axes[1].plot(times, [float(row["gpu_memory_percent"] or 0) for row in resources], label="GPU memory %")
        axes[2].plot(times, [float(row["gpu_temperature_c"] or 0) for row in resources], label="GPU °C")
        axes[2].plot(times, [float(row["gpu_power_w"] or 0) for row in resources], label="GPU W")
        axes[2].plot(times, [float(row.get("cpu_temperature_c") or 0) for row in resources], label="CPU °C")
        axes[2].plot(times, [float(row.get("cpu_power_w") or 0) for row in resources], label="CPU W")
        for axis in axes:
            axis.legend(loc="upper right")
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Elapsed seconds")
        fig.suptitle("Resource utilization")
        fig.tight_layout()
        target = plots_dir / "resources_over_time.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        generated.append(str(target.relative_to(run_dir)))
    return generated


def generate_comparison_plots(matrix_dir: Path) -> list[str]:
    plt = _plot_imports()
    summaries = []
    for path in sorted(matrix_dir.glob("*/summary.json")):
        data = json.loads(path.read_text())
        if data.get("status") == "complete":
            summaries.append(data)
    plots_dir = matrix_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    if not summaries:
        return generated

    mode_order = {"cpu": 0, "gpu": 1}
    summaries.sort(key=lambda item: (mode_order.get(item["mode"], 99), item["fps_per_camera"]))
    labels = [
        f"{item['mode'].upper()} {item['fps_per_camera']}\n({item['fps_per_camera'] * item['cameras']} total FPS)"
        for item in summaries
    ]
    colors = ["#4f81bd" if item["mode"] == "cpu" else "#59a14f" for item in summaries]
    x = list(range(len(summaries)))

    # One directly comparable latency chart containing every experiment.
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.25
    latency = [item["latencies_ms"]["end_to_end_ms"] for item in summaries]
    ax.bar([value - width for value in x], [item["average"] for item in latency], width, label="average")
    ax.bar(x, [item["p95"] for item in latency], width, label="p95")
    ax.bar([value + width for value in x], [item["p99"] for item in latency], width, label="p99")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Milliseconds", title="All experiments: end-to-end latency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = plots_dir / "all_experiments_latency.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))

    # Capacity curve: requested load, achieved throughput and dropped percentage.
    fig, (throughput_ax, drop_ax) = plt.subplots(2, 1, figsize=(11, 10))
    maximum_requested = max(item["fps_per_camera"] * item["cameras"] for item in summaries)
    throughput_ax.plot([0, maximum_requested], [0, maximum_requested], color="#888", linestyle="--", label="ideal")
    for mode in ("cpu", "gpu"):
        selected = [item for item in summaries if item["mode"] == mode]
        if not selected:
            continue
        requested = [item["fps_per_camera"] * item["cameras"] for item in selected]
        throughput_ax.plot(requested, [item["processed_fps"] for item in selected], marker="o", label=mode.upper())
        drop_ax.plot(requested, [item["dropped_percent"] for item in selected], marker="o", label=mode.upper())
    throughput_ax.set(ylabel="Processed FPS", title="Requested load versus achieved throughput")
    drop_ax.set(xlabel="Requested total FPS", ylabel="Dropped frames (%)", title="Drops at each load")
    for axis in (throughput_ax, drop_ax):
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    target = plots_dir / "capacity_and_drops.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))

    # P95 stage heatmap reveals the component that saturates first.
    stage_fields = [
        "decode_ms", "queue_wait_ms", "yolo_preprocess_ms", "yolo_inference_ms",
        "yolo_postprocess_ms", "crop_extraction_ms", "ocr_preprocess_ms",
        "ocr_inference_ms", "ocr_postprocess_ms", "rules_event_ms",
    ]
    stage_labels = [field.removesuffix("_ms").replace("_", " ") for field in stage_fields]
    stage_values = [
        [item["latencies_ms"].get(field, {}).get("p95", 0) for field in stage_fields]
        for item in summaries
    ]
    fig, ax = plt.subplots(figsize=(15, max(6, len(summaries) * 0.65)))
    image = ax.imshow(stage_values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(stage_labels)), stage_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("All experiments: p95 latency by pipeline stage (ms)")
    fig.colorbar(image, ax=ax, label="Milliseconds")
    for row_index, row in enumerate(stage_values):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    target = plots_dir / "stage_latency_heatmap.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))

    # Every experiment on one utilization chart.
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.22
    cpu = [item["resources"].get("cpu_percent", {}).get("average", 0) for item in summaries]
    gpu = [item["resources"].get("gpu_util_percent", {}).get("average", 0) for item in summaries]
    ram = [item["resources"].get("ram_percent", {}).get("average", 0) for item in summaries]
    ax.bar([value - width for value in x], cpu, width, label="CPU %")
    ax.bar(x, gpu, width, label="GPU %")
    ax.bar([value + width for value in x], ram, width, label="RAM %")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Average utilization (%)", title="All experiments: compute and memory utilization")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = plots_dir / "all_experiments_utilization.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))

    # Memory footprint shown separately because MB cannot share a percentage axis.
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.35
    process_ram = [item["resources"].get("process_ram_mb", {}).get("average", 0) for item in summaries]
    gpu_memory = [item["resources"].get("gpu_memory_mb", {}).get("average", 0) for item in summaries]
    ax.bar([value - width / 2 for value in x], process_ram, width, color=colors, alpha=0.75, label="Process RAM MB")
    ax.bar([value + width / 2 for value in x], gpu_memory, width, color="#e15759", alpha=0.75, label="GPU memory MB")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Megabytes", title="All experiments: memory footprint")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = plots_dir / "all_experiments_memory.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))

    # Thermal and power behavior across the complete matrix.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    thermal_fields = [
        ("cpu_temperature_c", "Average CPU temperature (°C)"),
        ("gpu_temperature_c", "Average GPU temperature (°C)"),
        ("cpu_power_w", "Average CPU power (W)"),
        ("gpu_power_w", "Average GPU power (W)"),
    ]
    for axis, (field, title) in zip(axes.flat, thermal_fields):
        values = [item["resources"].get(field, {}).get("average", 0) for item in summaries]
        axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    target = plots_dir / "all_experiments_thermal_power.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))
    return generated
