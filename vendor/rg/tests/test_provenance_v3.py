import json
from pathlib import Path
import tempfile
import unittest

from row_rgmap.analysis import file_sha256, seal_record
from row_rgmap.provenance_v3 import derive_seed
from row_rgmap.provenance_v3 import validate_qualification


class SeedDerivationTests(unittest.TestCase):
    def test_seed_derivation_is_deterministic_and_domain_separated(self):
        labels = ["trajectory:t0", "trajectory:t1", "qualification:null"]
        first = [derive_seed(2026090204, label) for label in labels]
        second = [derive_seed(2026090204, label) for label in labels]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))
        self.assertTrue(all(0 <= value < 2**32 for value in first))


class QualificationProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path):
        protocol = {"protocol_id": "unit-protocol"}
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        imported_source = root / "imported.py"
        imported_source.write_text("VALUE = 1\n", encoding="utf-8")
        record = seal_record(
            {
                "schema_version":
                    "pldr-row-rg-confirmation-v3-qualification-v1",
                "protocol_id": protocol["protocol_id"],
                "protocol_file_sha256": file_sha256(protocol_path),
                "all_launch_gates_passed": True,
                "gates": {"null": True, "power": True},
                "qualification_sources": [
                    {
                        "path": str(imported_source),
                        "sha256": file_sha256(imported_source),
                        "size_bytes": imported_source.stat().st_size,
                    }
                ],
            }
        )
        qualification_path = root / "qualification.json"
        qualification_path.write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )
        descriptor = {
            "path": str(qualification_path),
            "file_sha256": file_sha256(qualification_path),
            "record_sha256": record["record_sha256"],
        }
        return protocol, protocol_path, imported_source, record, descriptor

    def test_exact_passing_record_and_sources_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, protocol_path, _, record, descriptor = self._fixture(root)
            path, observed = validate_qualification(
                descriptor, root, protocol_path, protocol
            )
            self.assertEqual(path, Path(descriptor["path"]))
            self.assertEqual(observed, record)

    def test_mutated_imported_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, protocol_path, source, _, descriptor = self._fixture(root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "imported-source"):
                validate_qualification(descriptor, root, protocol_path, protocol)

    def test_missing_or_mutated_qualification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, protocol_path, _, _, descriptor = self._fixture(root)
            qualification = Path(descriptor["path"])
            qualification.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file digest"):
                validate_qualification(descriptor, root, protocol_path, protocol)
            qualification.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_qualification(descriptor, root, protocol_path, protocol)


if __name__ == "__main__":
    unittest.main()
