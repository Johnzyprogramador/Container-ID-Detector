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

    metrics = [
        ("latency", "End-to-end latency (ms)", lambda item: item["latencies_ms"]["end_to_end_ms"]),
        ("throughput", "Processed frames/second", lambda item: {"average": item["processed_fps"]}),
        ("drops", "Dropped frames (%)", lambda item: {"average": item["dropped_percent"]}),
    ]
    for filename, ylabel, getter in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        for mode in sorted({item["mode"] for item in summaries}):
            selected = sorted((item for item in summaries if item["mode"] == mode), key=lambda item: item["fps_per_camera"])
            fps = [item["fps_per_camera"] for item in selected]
            values = [getter(item)["average"] for item in selected]
            ax.plot(fps, values, marker="o", label=mode)
            if filename == "latency":
                ax.plot(fps, [getter(item)["p95"] for item in selected], marker="x", linestyle="--", label=f"{mode} p95")
                ax.plot(fps, [getter(item)["p99"] for item in selected], marker=".", linestyle=":", label=f"{mode} p99")
        ax.set(xlabel="FPS per camera", ylabel=ylabel, title=f"CPU vs GPU: {ylabel}")
        ax.set_xticks(sorted({item["fps_per_camera"] for item in summaries}))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = plots_dir / f"{filename}_comparison.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(str(target.relative_to(matrix_dir)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    resource_fields = [
        ("cpu_percent", "Average CPU %"), ("ram_percent", "Average RAM %"),
        ("gpu_util_percent", "Average GPU %"), ("gpu_memory_mb", "Average GPU memory MB"),
    ]
    for axis, (field, title) in zip(axes.flat, resource_fields):
        for mode in sorted({item["mode"] for item in summaries}):
            selected = sorted((item for item in summaries if item["mode"] == mode), key=lambda item: item["fps_per_camera"])
            axis.plot(
                [item["fps_per_camera"] for item in selected],
                [item["resources"].get(field, {}).get("average", 0) for item in selected],
                marker="o", label=mode,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    target = plots_dir / "resource_comparison.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    generated.append(str(target.relative_to(matrix_dir)))
    return generated
