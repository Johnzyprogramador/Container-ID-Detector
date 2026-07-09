import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from container_vision.field_capture.sessions import SessionStore, validate_id


class FieldCaptureTests(unittest.TestCase):
    def test_session_persists_frames_metrics_and_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary))
            session = store.create(logical_streams=2, device_name="test phone")
            result = {
                "detections": [{"class_name": "license_plate"}],
                "metrics": {
                    "sequence": 7,
                    "capture_time_ms": 10,
                    "server_receive_time_ms": 20,
                    "decode_ms": 1,
                    "inference_ms": 3,
                    "server_total_ms": 5,
                    "logical_streams": 2,
                    "detections": 1,
                    "jpeg_bytes": 4,
                    "client_skipped": 0,
                },
            }
            session.save_frame_result(7, b"jpeg", result)
            session.save_client_metric(
                {"sequence": 7, "client_receive_time_ms": 30, "round_trip_ms": 20}
            )
            session.save_segment(0, b"video", "mp4")
            session.stop()

            self.assertTrue((session.directory / "inference_frames/frame_00000007.jpg").is_file())
            self.assertTrue((session.directory / "results/frame_00000007.json").is_file())
            self.assertTrue((session.directory / "metrics/frames.csv").is_file())
            self.assertTrue((session.directory / "metrics/client.csv").is_file())
            self.assertTrue((session.directory / "cloud_recording/segment_00000.mp4").is_file())
            metadata = json.loads((session.directory / "session.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["frames_received"], 1)
            self.assertEqual(metadata["segments_received"], 1)

    def test_session_archive_contains_capture_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = SessionStore(Path(temporary)).create(logical_streams=1)
            session.save_segment(0, b"video", "mp4")
            session.save_frame_result(
                1,
                b"jpeg",
                {
                    "detections": [],
                    "metrics": {
                        "sequence": 1,
                        "capture_time_ms": 10,
                        "server_receive_time_ms": 20,
                        "decode_ms": 1,
                        "inference_ms": 0,
                        "server_total_ms": 2,
                        "logical_streams": 1,
                        "detections": 0,
                        "jpeg_bytes": 4,
                        "client_skipped": 0,
                    },
                },
            )
            archive_path = session.archive()

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())

            self.assertIn("session.json", names)
            self.assertIn("cloud_recording/segment_00000.mp4", names)
            self.assertIn("inference_frames/frame_00000001.jpg", names)
            self.assertIn("results/frame_00000001.json", names)
            self.assertIn("metrics/frames.csv", names)

    def test_rejects_invalid_multiplier(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                SessionStore(Path(temporary)).create(logical_streams=4)

    def test_rejects_path_like_session_id(self):
        with self.assertRaises(ValueError):
            validate_id("../outside", "session id")
