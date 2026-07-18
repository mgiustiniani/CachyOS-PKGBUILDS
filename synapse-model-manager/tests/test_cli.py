from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class ResumableHandler(BaseHTTPRequestHandler):
    payload = b""
    fail_first = True
    ranges: list[int] = []
    delay = 0.0
    chunk_size = 64 * 1024

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        range_value = self.headers.get("Range")
        offset = int(range_value.removeprefix("bytes=").split("-", 1)[0]) if range_value else 0
        type(self).ranges.append(offset)
        body = type(self).payload[offset:]
        self.send_response(206 if offset else 200)
        self.send_header("Content-Length", str(len(body)))
        if offset:
            self.send_header("Content-Range", f"bytes {offset}-{len(type(self).payload) - 1}/{len(type(self).payload)}")
        self.end_headers()
        if type(self).fail_first and not offset:
            type(self).fail_first = False
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.shutdown(1)
            return
        if type(self).delay:
            try:
                for start in range(0, len(body), type(self).chunk_size):
                    self.wfile.write(body[start : start + type(self).chunk_size])
                    self.wfile.flush()
                    time.sleep(type(self).delay)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.wfile.write(body)


class ModelManagerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifests = self.root / "manifests"
        self.source = self.root / "usb" / "models"
        self.destination = self.root / "destination"
        self.state = self.root / "state"
        self.manifests.mkdir(parents=True)
        model_root = self.source / "safetensors/test/model"
        model_root.mkdir(parents=True)
        payload = b"synapse-model-fixture\n"
        (model_root / "weights.bin").write_bytes(payload)
        (model_root / "config.json").write_text("{}\n", encoding="utf-8")
        (model_root / ".snapshot-revision").write_text("deadbeef\n", encoding="utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        (self.manifests / "fixture.toml").write_text(
            f'''schema_version = 1
id = "fixture"
name = "Fixture"
description = "Test model"
product = "tests"
default_root = "{self.destination}"
license = "MIT"

[[components]]
id = "model"
relative_path = "safetensors/test/model"
source_type = "huggingface"
repo = "example/fixture"
revision = "deadbeef"
required = [
  {{ path = "weights.bin", size = {len(payload)}, sha256 = "{digest}" }},
  {{ path = "config.json" }},
]
''',
            encoding="utf-8",
        )
        web_source = self.root / "web-source.bin"
        web_source.write_bytes(b"web-model-fixture\n")
        web_hash = hashlib.sha256(web_source.read_bytes()).hexdigest()
        (self.manifests / "web-fixture.toml").write_text(
            f'''schema_version = 1
id = "web-fixture"
name = "Web Fixture"
description = "Web acquisition test"
product = "tests"
default_root = "{self.destination}"
license = "MIT"

[[components]]
id = "web-model"
relative_path = "gguf/test/web"
source_type = "http-files"
revision = "web-revision"
required = [
  {{ path = "model.bin", size = {web_source.stat().st_size}, sha256 = "{web_hash}" }},
]
http_files = [
  {{ path = "model.bin", url = "{web_source.as_uri()}", size = {web_source.stat().st_size}, sha256 = "{web_hash}" }},
]
''',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def cli_command(self, *arguments: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "synapse_model_manager.cli",
            "--manifest-dir",
            str(self.manifests),
            "--state-dir",
            str(self.state),
            *arguments,
        ]

    def run_cli(self, *arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT)
        result = subprocess.run(
            self.cli_command(*arguments),
            cwd=PROJECT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, expected, result.stderr + result.stdout)
        return result

    def test_json_list_is_stable_envelope(self) -> None:
        result = self.run_cli("list", "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schemaVersion"], "1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["models"][0]["id"], "fixture")
        self.assertEqual(result.stderr, "")

    def test_external_install_verify_and_resolve(self) -> None:
        installed = self.run_cli(
            "install", "fixture", "--source", str(self.source), "--mode", "external", "--json"
        )
        payload = json.loads(installed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["mode"], "external")
        self.assertEqual((self.state / "activations.json").stat().st_mode & 0o777, 0o644)
        resolved = json.loads(self.run_cli("resolve", "fixture", "--json").stdout)
        self.assertEqual(
            resolved["data"]["components"][0]["path"],
            str(self.source / "safetensors/test/model"),
        )

    def test_copy_install_is_verified_and_activated(self) -> None:
        result = self.run_cli(
            "install", "fixture", "--source", str(self.source), "--mode", "copy",
            "--root", str(self.destination), "--json",
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["data"]["validation"]["valid"])
        self.assertEqual(
            (self.destination / "safetensors/test/model/weights.bin").read_bytes(),
            b"synapse-model-fixture\n",
        )
        status = json.loads(self.run_cli("status", "fixture", "--json").stdout)
        self.assertTrue(status["data"]["models"][0]["validation"]["valid"])

    def test_copy_install_resumes_persistent_partial(self) -> None:
        partial = self.destination / ".synapse-model-staging/fixture/safetensors/test/model/weights.bin.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"synapse-")
        result = self.run_cli(
            "install", "fixture", "--source", str(self.source), "--mode", "copy",
            "--root", str(self.destination), "--jsonl",
        )
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(events[-1]["event"], "result")
        self.assertEqual(
            (self.destination / "safetensors/test/model/weights.bin").read_bytes(),
            b"synapse-model-fixture\n",
        )

    def test_checksum_failure_is_structured(self) -> None:
        (self.source / "safetensors/test/model/weights.bin").write_bytes(b"x" * len(b"synapse-model-fixture\n"))
        result = self.run_cli("verify", "fixture", "--source", str(self.source), "--json", expected=4)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["errors"][0]["code"], "validation_failed")

    def test_web_install_downloads_verifies_and_activates(self) -> None:
        result = self.run_cli(
            "install", "web-fixture", "--source", "web", "--mode", "copy",
            "--root", str(self.destination), "--json",
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["source"], "web")
        self.assertEqual(
            (self.destination / "gguf/test/web/model.bin").read_bytes(),
            b"web-model-fixture\n",
        )

    def test_http_failure_preserves_partial_and_second_run_resumes(self) -> None:
        ResumableHandler.payload = (b"0123456789abcdef" * 262144) + b"end"
        ResumableHandler.fail_first = True
        ResumableHandler.ranges = []
        ResumableHandler.delay = 0.0
        server = ThreadingHTTPServer(("127.0.0.1", 0), ResumableHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            digest = hashlib.sha256(ResumableHandler.payload).hexdigest()
            port = server.server_address[1]
            (self.manifests / "resume.toml").write_text(
                f'''schema_version = 1
id = "resume-fixture"
name = "Resume Fixture"
description = "Persistent HTTP resume test"
product = "tests"
default_root = "{self.destination}"
license = "MIT"

[[components]]
id = "model"
relative_path = "gguf/test/resume"
source_type = "http-files"
revision = "resume-revision"
required = [
  {{ path = "model.bin", size = {len(ResumableHandler.payload)}, sha256 = "{digest}" }},
]
http_files = [
  {{ path = "model.bin", url = "http://127.0.0.1:{port}/model.bin", size = {len(ResumableHandler.payload)}, sha256 = "{digest}" }},
]
''',
                encoding="utf-8",
            )
            first = self.run_cli(
                "install", "resume-fixture", "--source", "web", "--mode", "copy",
                "--root", str(self.destination), "--json", expected=5,
            )
            self.assertEqual(json.loads(first.stdout)["errors"][0]["code"], "acquisition_failed")
            partial = self.destination / ".synapse-model-staging/resume-fixture/gguf/test/resume/model.bin.part"
            self.assertGreater(partial.stat().st_size, 0)
            failed_job = json.loads((self.state / "jobs/resume-fixture.json").read_text())
            self.assertEqual(failed_job["state"], "failed")

            second = self.run_cli(
                "install", "resume-fixture", "--source", "web", "--mode", "copy",
                "--root", str(self.destination), "--jsonl",
            )
            events = [json.loads(line) for line in second.stdout.splitlines()]
            starts = [event for event in events if event["event"] == "download-started"]
            self.assertGreater(starts[0]["resumeOffsetBytes"], 0)
            self.assertGreater(ResumableHandler.ranges[-1], 0)
            self.assertEqual(
                (self.destination / "gguf/test/resume/model.bin").read_bytes(),
                ResumableHandler.payload,
            )
            jobs = json.loads(self.run_cli("jobs", "resume-fixture", "--json").stdout)
            self.assertEqual(jobs["data"]["jobs"][0]["state"], "completed")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_active_http_job_can_be_cancelled_and_resumed(self) -> None:
        ResumableHandler.payload = b"x" * (16 * 1024 * 1024)
        ResumableHandler.fail_first = False
        ResumableHandler.ranges = []
        ResumableHandler.delay = 0.005
        server = ThreadingHTTPServer(("127.0.0.1", 0), ResumableHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            digest = hashlib.sha256(ResumableHandler.payload).hexdigest()
            port = server.server_address[1]
            (self.manifests / "cancel.toml").write_text(
                f'''schema_version = 1
id = "cancel-fixture"
name = "Cancel Fixture"
description = "Cooperative cancellation test"
product = "tests"
default_root = "{self.destination}"
license = "MIT"

[[components]]
id = "model"
relative_path = "gguf/test/cancel"
source_type = "http-files"
required = [
  {{ path = "model.bin", size = {len(ResumableHandler.payload)}, sha256 = "{digest}" }},
]
http_files = [
  {{ path = "model.bin", url = "http://127.0.0.1:{port}/model.bin", size = {len(ResumableHandler.payload)}, sha256 = "{digest}" }},
]
''',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT)
            process = subprocess.Popen(
                self.cli_command(
                    "install", "cancel-fixture", "--source", "web", "--mode", "copy",
                    "--root", str(self.destination), "--json",
                ),
                cwd=PROJECT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            job_path = self.state / "jobs/cancel-fixture.json"
            partial = self.destination / ".synapse-model-staging/cancel-fixture/gguf/test/cancel/model.bin.part"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not partial.is_file():
                time.sleep(0.02)
            self.assertTrue(job_path.is_file())
            self.assertTrue(partial.is_file())
            conflict = self.run_cli(
                "install", "cancel-fixture", "--source", "web", "--mode", "copy",
                "--root", str(self.destination), "--json", expected=6,
            )
            self.assertEqual(json.loads(conflict.stdout)["errors"][0]["code"], "job_conflict")
            cancelled = self.run_cli("cancel", "cancel-fixture", "--json")
            self.assertTrue(json.loads(cancelled.stdout)["ok"])
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 7, stderr + stdout)
            self.assertEqual(json.loads(stdout)["errors"][0]["code"], "job_cancelled")
            self.assertGreater(partial.stat().st_size, 0)
            self.assertEqual(json.loads(job_path.read_text())["state"], "cancelled")

            ResumableHandler.delay = 0.0
            resumed = self.run_cli("resume", "cancel-fixture", "--json")
            self.assertTrue(json.loads(resumed.stdout)["ok"])
            self.assertGreater(ResumableHandler.ranges[-1], 0)
            self.assertEqual(json.loads(job_path.read_text())["state"], "completed")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            ResumableHandler.delay = 0.0

    def test_unfinished_job_parameter_change_requires_restart(self) -> None:
        result = self.run_cli(
            "install", "web-fixture", "--source", "usb", "--mode", "copy",
            "--root", str(self.destination), "--json", expected=3,
        )
        self.assertFalse(json.loads(result.stdout)["ok"])
        other = self.root / "other-destination"
        conflict = self.run_cli(
            "install", "web-fixture", "--source", "web", "--mode", "copy",
            "--root", str(other), "--json", expected=6,
        )
        self.assertEqual(json.loads(conflict.stdout)["errors"][0]["code"], "job_conflict")
        restarted = self.run_cli(
            "install", "web-fixture", "--source", "web", "--mode", "copy",
            "--root", str(other), "--restart", "--json",
        )
        self.assertTrue(json.loads(restarted.stdout)["ok"])

    def test_background_queue_runs_highest_priority_first(self) -> None:
        low = self.run_cli(
            "enqueue", "fixture", "--source", str(self.source), "--mode", "copy",
            "--root", str(self.destination), "--priority", "1", "--no-start-worker", "--json",
        )
        high = self.run_cli(
            "enqueue", "web-fixture", "--source", "web", "--mode", "copy",
            "--root", str(self.destination), "--priority", "10", "--no-start-worker", "--json",
        )
        self.assertEqual(json.loads(low.stdout)["data"]["job"]["state"], "queued")
        self.assertFalse(json.loads(high.stdout)["data"]["workerStarted"])
        worker = json.loads(self.run_cli("worker", "--max-jobs", "1", "--json").stdout)
        self.assertEqual(worker["data"]["attempts"][0]["model"], "web-fixture")
        states = {job["model"]: job["state"] for job in json.loads(self.run_cli("jobs", "--json").stdout)["data"]["jobs"]}
        self.assertEqual(states["web-fixture"], "completed")
        self.assertEqual(states["fixture"], "queued")

    def test_worker_recovers_stale_running_job(self) -> None:
        self.run_cli(
            "enqueue", "web-fixture", "--source", "web", "--mode", "copy",
            "--root", str(self.destination), "--no-start-worker", "--json",
        )
        job_path = self.state / "jobs/web-fixture.json"
        job = json.loads(job_path.read_text())
        job["state"] = "running"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        result = json.loads(self.run_cli("worker", "--json").stdout)
        self.assertTrue(result["data"]["attempts"][0]["ok"])
        self.assertEqual(json.loads(job_path.read_text())["state"], "completed")

    def test_retry_and_remove_job_with_partials(self) -> None:
        self.run_cli(
            "install", "web-fixture", "--source", "usb", "--mode", "copy",
            "--root", str(self.destination), "--json", expected=3,
        )
        retried = json.loads(self.run_cli("retry", "web-fixture", "--json").stdout)
        self.assertEqual(retried["data"]["job"]["state"], "queued")
        staging = self.destination / ".synapse-model-staging/web-fixture"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "partial").write_bytes(b"partial")
        removed = json.loads(self.run_cli("remove-job", "web-fixture", "--partials", "--json").stdout)
        self.assertTrue(removed["data"]["partialsRemoved"])
        self.assertFalse(staging.exists())
        self.assertEqual(json.loads(self.run_cli("jobs", "web-fixture", "--json").stdout)["data"]["jobs"], [])

    def test_jsonl_emits_events_and_final_result(self) -> None:
        result = self.run_cli(
            "install", "fixture", "--source", str(self.source), "--mode", "copy",
            "--root", str(self.destination), "--jsonl",
        )
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertIn("copy-started", [entry["event"] for entry in events])
        self.assertEqual(events[-1]["event"], "result")
        self.assertTrue(events[-1]["ok"])


if __name__ == "__main__":
    unittest.main()
