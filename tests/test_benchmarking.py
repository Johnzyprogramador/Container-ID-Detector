from __future__ import annotations

import unittest
from pathlib import Path

from container_vision.benchmarking import percentile, summarize_rows
from scripts.run_benchmarks import EventDecider, assign_camera_playlists


class BenchmarkingTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_summarize_rows(self) -> None:
        summary = summarize_rows([{"latency": 10}, {"latency": 20}], ["latency"])
        self.assertEqual(summary["latency"]["average"], 15)
        self.assertEqual(summary["latency"]["maximum"], 20)

    def test_video_assignment_is_round_robin(self) -> None:
        videos = [Path(f"video_{index}.mp4") for index in range(6)]
        playlists = assign_camera_playlists(videos)
        self.assertEqual([len(items) for items in playlists], [2, 2, 1, 1])

    def test_event_decider_emits_after_required_observations(self) -> None:
        decider = EventDecider(observations=3, timeout_s=10)
        self.assertIsNone(decider.add(0, "license_plate", "ABC", 0.8, 1.0))
        self.assertIsNone(decider.add(0, "license_plate", "ABC", 0.9, 1.1))
        event = decider.add(0, "license_plate", "A8C", 0.1, 1.2)
        self.assertEqual(event["decision"], "ABC")
        self.assertEqual(event["observations"], 3)


if __name__ == "__main__":
    unittest.main()
