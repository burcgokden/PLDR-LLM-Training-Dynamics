from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256
from scripts.analyze_energy_closure_probe_v3 import (
    analyze,
    verify_output_checkpoint_inventory,
)
from scripts.verify_energy_closure_relocation_v3 import (
    validate_source_checkpoint_census,
)


class EnergyClosureV3CheckpointContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_root = Path(self.temporary.name).resolve()
        self.branch_root = self.run_root / "trajectories" / "T" / "B"
        self.checkpoint_root = self.branch_root / "checkpoints"
        self.checkpoint_root.mkdir(parents=True)
        self.files = {}
        for step, payload in ((0, b"source"), (24, b"successor")):
            path = self.checkpoint_root / f"checkpoint-step{step:06d}.pt"
            path.write_bytes(payload)
            self.files[step] = (path, payload)
        self.metadata = {
            "checkpoints": [
                {
                    "path": path.relative_to(self.branch_root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "step": step,
                }
                for step, (path, _payload) in sorted(self.files.items())
            ]
        }
        self.row = {
            "output_checkpoints": [
                {
                    "trajectory_id": "T",
                    "branch": "B",
                    "step": item["step"],
                    "path": (self.branch_root / item["path"])
                    .relative_to(self.run_root)
                    .as_posix(),
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
                for item in self.metadata["checkpoints"]
            ]
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verify(self, row=None, metadata=None):
        return verify_output_checkpoint_inventory(
            self.run_root,
            self.row if row is None else row,
            self.metadata if metadata is None else metadata,
            "T",
            "B",
            {0, 24},
        )

    def test_exact_inventory_replays(self) -> None:
        count, size = self.verify()
        self.assertEqual(count, 2)
        self.assertEqual(size, len(b"source") + len(b"successor"))

    def test_each_checkpoint_tamper_is_rejected(self) -> None:
        for step, (path, original) in self.files.items():
            with self.subTest(step=step):
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "descriptor fails"):
                    self.verify()
                path.write_bytes(original)

    def test_missing_descriptor_is_rejected(self) -> None:
        row = copy.deepcopy(self.row)
        row["output_checkpoints"].pop()
        with self.assertRaisesRegex(ValueError, "exactly bound"):
            self.verify(row=row)

    def test_extra_file_is_rejected(self) -> None:
        (self.checkpoint_root / "undeclared.pt").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "exactly bound"):
            self.verify()

    def test_path_escape_is_rejected(self) -> None:
        row = copy.deepcopy(self.row)
        row["output_checkpoints"][0]["path"] = "../outside.pt"
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            self.verify(row=row)

    def relocation_manifest(self):
        return {
            "branches": {"T": [{"branch": "B", **copy.deepcopy(self.row)}]},
            "checkpoint_inventory": {
                "descriptor_count": len(self.row["output_checkpoints"]),
                "total_size_bytes": sum(
                    item["size_bytes"] for item in self.row["output_checkpoints"]
                ),
                "all_emitted_files_bound": True,
            },
        }

    def test_relocation_source_census_is_exact(self) -> None:
        census = validate_source_checkpoint_census(
            self.run_root, self.relocation_manifest()
        )
        self.assertTrue(census["exact"])
        self.assertEqual(census["checkpoint_count"], 2)

    def test_relocation_source_census_rejects_extra_missing_changed_and_escape(self) -> None:
        extra = self.checkpoint_root / "extra.pt"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "census is not exact"):
            validate_source_checkpoint_census(
                self.run_root, self.relocation_manifest()
            )
        extra.unlink()

        undeclared_root = (
            self.run_root / "trajectories" / "T" / "undeclared" / "checkpoints"
        )
        undeclared_root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "checkpoint-directory census"):
            validate_source_checkpoint_census(
                self.run_root, self.relocation_manifest()
            )
        undeclared_root.rmdir()
        undeclared_root.parent.rmdir()

        missing = self.files[24][0]
        original = missing.read_bytes()
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "missing or a symlink"):
            validate_source_checkpoint_census(
                self.run_root, self.relocation_manifest()
            )
        missing.write_bytes(original)

        changed = self.files[0][0]
        original = changed.read_bytes()
        changed.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "descriptor fails"):
            validate_source_checkpoint_census(
                self.run_root, self.relocation_manifest()
            )
        changed.write_bytes(original)

        escaped = self.relocation_manifest()
        escaped["branches"]["T"][0]["output_checkpoints"][0][
            "path"
        ] = "../outside.pt"
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_source_checkpoint_census(self.run_root, escaped)

    def test_run_root_outside_data_root_fails_before_read(self) -> None:
        outside = self.run_root / "outside"
        data_root = self.run_root / "data"
        outside.mkdir()
        data_root.mkdir()
        with self.assertRaisesRegex(ValueError, "inside the data root"):
            analyze(outside, data_root, commit="0" * 40, sources=[])


if __name__ == "__main__":
    unittest.main()
