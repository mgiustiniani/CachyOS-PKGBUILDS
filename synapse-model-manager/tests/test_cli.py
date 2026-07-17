from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


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

    def run_cli(self, *arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "synapse_model_manager.cli",
                "--manifest-dir",
                str(self.manifests),
                "--state-dir",
                str(self.state),
                *arguments,
            ],
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
