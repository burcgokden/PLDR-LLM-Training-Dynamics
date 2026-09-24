import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from row_rgmap.analysis import file_sha256, seal_record
from row_rgmap.provenance_v3 import git_blob_descriptor
from scripts.verify_registered_artifacts import (
    event_rows,
    latest_root_manifest_path,
    root_manifest_chain,
    verify_analysis_source_registry,
    verify_record,
)
from scripts.run_confirmation_v3 import _append_event


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class RegisteredArtifactVerifierTests(unittest.TestCase):
    def test_non_ascii_record_uses_writer_canonical_encoding(self):
        record = seal_record({"label": "universality γ"})
        verify_record(record)

    def test_latest_complete_manifest_requires_every_higher_entry_to_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_digest = None
            for index in (1, 2):
                result = seal_record({"analysis_id": f"analysis-{index:04d}"})
                result_path = root / "analyses" / f"analysis-{index:04d}" / "result.json"
                _write(result_path, result)
                analysis = seal_record(
                    {
                        "result": {
                            "path": str(result_path.relative_to(root)),
                            "sha256": file_sha256(result_path),
                            "record_sha256": result["record_sha256"],
                        }
                    }
                )
                analysis_path = result_path.parent / "manifest.json"
                _write(analysis_path, analysis)
                manifest = seal_record(
                    {
                        "schema_version": "pldr-row-rg-root-manifest-v3.1",
                        "manifest_id": f"manifest-{index:04d}",
                        "previous_manifest_record_sha256": previous_digest,
                        "analysis_manifest": {
                            "path": str(analysis_path.relative_to(root)),
                            "sha256": file_sha256(analysis_path),
                            "record_sha256": analysis["record_sha256"],
                        },
                    }
                )
                _write(root / "manifests" / f"manifest-{index:04d}.json", manifest)
                previous_digest = manifest["record_sha256"]
            self.assertEqual(
                latest_root_manifest_path(root).name, "manifest-0002.json"
            )
            (root / "manifests" / "manifest-0003.json").write_text("{}\n")
            with self.assertRaisesRegex(
                ValueError, "invalid root manifest candidate manifest-0003"
            ):
                latest_root_manifest_path(root)

    def test_verifier_event_chain_detects_broken_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            _append_event(path, "FIRST", {"label": "γ"})
            _append_event(path, "SECOND", {"value": 2})
            rows = path.read_text(encoding="utf-8").splitlines()
            second = json.loads(rows[1])
            second["previous_event_sha256"] = "0" * 64
            rows[1] = json.dumps(second, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parent digest"):
                event_rows(path)

    def test_root_manifest_chain_replays_every_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_directory = root / "manifests"
            previous_digest = None
            paths = []
            for index in range(1, 4):
                manifest = seal_record(
                    {
                        "schema_version": "pldr-row-rg-root-manifest-v3",
                        "manifest_id": f"manifest-{index:04d}",
                        "previous_manifest_record_sha256": previous_digest,
                    }
                )
                path = manifest_directory / f"manifest-{index:04d}.json"
                _write(path, manifest)
                paths.append(path)
                previous_digest = manifest["record_sha256"]
            self.assertEqual(
                [row["manifest_id"] for row in root_manifest_chain(root, paths[-1])],
                ["manifest-0001", "manifest-0002", "manifest-0003"],
            )

            second = json.loads(paths[1].read_text(encoding="utf-8"))
            second.pop("record_sha256")
            second["previous_manifest_record_sha256"] = "0" * 64
            _write(paths[1], seal_record(second))
            with self.assertRaisesRegex(ValueError, "parent digest"):
                root_manifest_chain(root, paths[-1])

    def test_analysis_manifest_source_registry_is_verified_independently(self):
        repository_root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        repository_path = "src/row_rgmap/analysis.py"
        digest, size = git_blob_descriptor(
            repository_root, commit, repository_path
        )
        row = {
            "repository_path": repository_path,
            "code_commit": commit,
            "sha256": digest,
            "size_bytes": size,
        }
        result = {
            "analyzer_code_commit": commit,
            "analysis_sources": [row],
        }
        manifest = {
            "analyzer_code_commit": commit,
            "analysis_sources": [dict(row)],
        }
        self.assertEqual(verify_analysis_source_registry(result, manifest), 1)
        manifest["analysis_sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValueError, "source registry differs from the result"
        ):
            verify_analysis_source_registry(result, manifest)


if __name__ == "__main__":
    unittest.main()
