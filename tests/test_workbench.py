from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from apps.workbench.app import (
    WorkbenchConfig,
    annotation_from_payload,
    annotation_path_for,
    list_field_capture_sessions,
    list_images,
    list_benchmark_matrices,
    list_prediction_videos,
    list_sessions,
    safe_relative_path,
    session_id_from_image_id,
)


class WorkbenchTests(unittest.TestCase):
    def test_list_images_recursively(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.jpg").write_bytes(b"")
            (root / "nested" / "b.png").write_bytes(b"")
            (root / "notes.txt").write_text("skip")

            images = list_images(root)

            self.assertEqual([item["id"] for item in images], ["a.jpg", "nested/b.png"])

    def test_list_sessions_from_frames_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "a.jpg").write_bytes(b"")
            (root / "session_002").mkdir()
            (root / "session_002" / "b.png").write_bytes(b"")
            (root / "empty_session").mkdir()

            sessions = list_sessions(root)

            self.assertEqual([item["id"] for item in sessions], ["session_001", "session_002"])
            self.assertEqual([item["image_count"] for item in sessions], [1, 1])

    def test_list_images_for_selected_session(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "a.jpg").write_bytes(b"")
            (root / "session_002").mkdir()
            (root / "session_002" / "b.png").write_bytes(b"")

            images = list_images(root, session_id="session_002")

            self.assertEqual([item["id"] for item in images], ["session_002/b.png"])
            self.assertEqual([item["path"] for item in images], ["b.png"])

    def test_list_prediction_videos_by_session(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "truck_predicted.mp4").write_bytes(b"")
            (root / "summary.json").write_text("{}")

            videos = list_prediction_videos(root)

            self.assertEqual(len(videos), 1)
            self.assertEqual(videos[0]["session"], "session_001")
            self.assertEqual(videos[0]["url"], "/prediction-media/session_001/truck_predicted.mp4")

    def test_list_benchmark_matrices_includes_live_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "matrix_001"
            run = matrix / "gpu_1fps"
            run.mkdir(parents=True)
            (matrix / "matrix.json").write_text('{"status": "running"}')
            (run / "live.json").write_text('{"status": "running", "processed_frames": 20}')

            matrices = list_benchmark_matrices(root)

            self.assertEqual(matrices[0]["id"], "matrix_001")
            self.assertEqual(matrices[0]["runs"][0]["name"], "gpu_1fps")
            self.assertEqual(matrices[0]["runs"][0]["live"]["processed_frames"], 20)

    def test_list_field_capture_sessions_summarizes_metrics_and_media(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "capture_001"
            (session / "cloud_recording").mkdir(parents=True)
            (session / "inference_frames").mkdir()
            (session / "metrics").mkdir()
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": "capture_001",
                        "status": "complete",
                        "logical_streams": 2,
                        "bytes_received": 1024,
                    }
                )
            )
            (session / "cloud_recording" / "segment_00000.mp4").write_bytes(b"video")
            (session / "inference_frames" / "frame_00000001.jpg").write_bytes(b"jpeg")
            (session / "metrics" / "frames.csv").write_text(
                "sequence,server_total_ms,decode_ms,inference_ms,jpeg_bytes,client_skipped,detections\n"
                "1,10,2,0,100,0,0\n"
                "2,20,3,0,120,1,0\n"
            )
            (session / "metrics" / "client.csv").write_text(
                "sequence,client_receive_time_ms,round_trip_ms\n"
                "1,1000,50\n"
                "2,1100,70\n"
            )

            captures = list_field_capture_sessions(root)

            self.assertEqual(captures[0]["id"], "capture_001")
            self.assertEqual(captures[0]["videos"][0]["url"], "/field-capture-media/capture_001/cloud_recording/segment_00000.mp4")
            self.assertEqual(captures[0]["frames"][0]["url"], "/field-capture-media/capture_001/inference_frames/frame_00000001.jpg")
            self.assertEqual(captures[0]["metrics"]["frame_rows"], 2)
            self.assertEqual(captures[0]["metrics"]["client_rows"], 2)
            self.assertEqual(captures[0]["metrics"]["total_client_skipped"], 1)
            self.assertEqual(captures[0]["summaries"]["round_trip_ms"]["average"], 60)
            self.assertAlmostEqual(captures[0]["summaries"]["server_total_ms"]["p99"], 19.9)

    def test_safe_relative_path_rejects_traversal(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                safe_relative_path(Path(tmp), "../outside.jpg")

    def test_annotation_path_preserves_nested_image_path(self) -> None:
        with TemporaryDirectory() as tmp:
            path = annotation_path_for(Path(tmp), "session/a.jpg")

            self.assertEqual(path, (Path(tmp) / "session" / "a.jpg.json").resolve())

    def test_payload_builds_valid_annotation(self) -> None:
        config = WorkbenchConfig(
            images_dir=Path("data/frames"),
            annotations_dir=Path("data/annotations"),
            session_id=None,
            host="127.0.0.1",
            port=7860,
        )

        annotation = annotation_from_payload(
            config,
            {
                "image_id": "session_001/truck.jpg",
                "width": 100,
                "height": 80,
                "objects": [
                    {
                        "bbox_xyxy": [10, 20, 40, 50],
                        "transcription": "274",
                        "readable": True,
                        "occluded": False,
                    }
                ],
            },
        )

        self.assertEqual(annotation.session_id, "session_001")
        self.assertEqual(annotation.objects[0].class_name, "painted_number")
        self.assertEqual(annotation.objects[0].transcription, "274")

    def test_payload_preserves_license_plate_class(self) -> None:
        config = WorkbenchConfig(
            images_dir=Path("data/frames"),
            annotations_dir=Path("data/annotations"),
            session_id=None,
            host="127.0.0.1",
            port=7860,
        )

        annotation = annotation_from_payload(
            config,
            {
                "image_id": "session_plate/car.jpg",
                "width": 100,
                "height": 80,
                "objects": [
                    {
                        "class_name": "license_plate",
                        "bbox_xyxy": [10, 20, 70, 50],
                        "transcription": "12AB34",
                        "readable": True,
                    }
                ],
            },
        )

        annotation.validate()
        self.assertEqual(annotation.objects[0].class_name, "license_plate")
        self.assertEqual(annotation.objects[0].transcription, "12AB34")

    def test_forced_session_id_overrides_image_path_session(self) -> None:
        config = WorkbenchConfig(
            images_dir=Path("data/frames/session_001"),
            annotations_dir=Path("data/annotations/session_001"),
            session_id="session_001",
            host="127.0.0.1",
            port=7860,
        )

        self.assertEqual(session_id_from_image_id(config, "truck.jpg"), "session_001")


if __name__ == "__main__":
    unittest.main()
