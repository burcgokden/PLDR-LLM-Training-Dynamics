from pathlib import Path
import tempfile
import unittest

from scripts.create_evidence_index import (
    build_executed_inventory,
    deterministic_tree_census,
)


class ExecutedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.executed = Path(self.temporary.name) / "executed"
        self.executed.mkdir()
        (self.executed / "current").mkdir()
        (self.executed / "current" / "result.json").write_text("{}\n")
        (self.executed / "failed").mkdir()
        (self.executed / "failed" / "events.jsonl").write_text("event\n")
        self.entries = {
            "current-result": {"path": "executed/current/result.json"},
            "failed-events": {"path": "executed/failed/events.jsonl"},
        }
        self.rows = [
            {
                "name": "current",
                "classification": "current-evidence",
                "role": "selected result",
                "representative_entry_ids": ["current-result"],
                "sealed_chain_verification": True,
            },
            {
                "name": "failed",
                "classification": "failed-launch",
                "role": "operational failure",
                "reason": "producer rejected the input",
                "representative_entry_ids": ["failed-events"],
                "tree_census": True,
            },
        ]

    def test_inventory_is_exact_and_failed_tree_is_content_addressed(self):
        inventory = build_executed_inventory(
            self.executed, self.rows, self.entries
        )
        failed = next(row for row in inventory if row["name"] == "failed")
        self.assertEqual(failed["tree_census"]["file_count"], 1)
        self.assertEqual(
            failed["verification_coverage_modes"],
            ["representative-file-digest", "recursive-tree-census"],
        )
        current = next(row for row in inventory if row["name"] == "current")
        self.assertEqual(
            current["verification_coverage_modes"],
            ["representative-file-digest", "internal-sealed-chain-verification"],
        )
        before = failed["tree_census"]["tree_sha256"]
        (self.executed / "failed" / "events.jsonl").write_text("changed\n")
        self.assertNotEqual(
            before,
            deterministic_tree_census(self.executed / "failed")["tree_sha256"],
        )

    def test_unclassified_or_declared_missing_child_is_rejected(self):
        (self.executed / "unclassified").mkdir()
        with self.assertRaisesRegex(ValueError, "unclassified"):
            build_executed_inventory(self.executed, self.rows, self.entries)
        (self.executed / "unclassified").rmdir()
        (self.executed / "current" / "result.json").unlink()
        (self.executed / "current").rmdir()
        with self.assertRaisesRegex(ValueError, "missing"):
            build_executed_inventory(self.executed, self.rows, self.entries)


if __name__ == "__main__":
    unittest.main()
