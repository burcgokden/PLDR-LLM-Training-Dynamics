import json
from pathlib import Path
import unittest
from unittest import mock

from scripts import check_formal_soundness


class FormalBuildBoundaryTests(unittest.TestCase):
    def test_public_target_depends_on_lean_build(self):
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(
            encoding="utf-8"
        )
        self.assertIn("formal-audit: lean-test", makefile)

    @mock.patch("scripts.check_formal_soundness.subprocess.run")
    def test_build_failure_prevents_artifact_audit(self, run):
        run.return_value = mock.Mock(returncode=1, stdout="", stderr="changed source")
        with self.assertRaisesRegex(RuntimeError, "build failed before"):
            check_formal_soundness.build_lean("lake")

    @mock.patch("scripts.check_formal_soundness.subprocess.run")
    def test_successful_build_report_binds_toolchain(self, run):
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        report = check_formal_soundness.build_lean("lake")
        self.assertTrue(report["completed"])
        self.assertEqual(report["command"], ["lake", "build"])
        self.assertEqual(
            report["toolchain"],
            (Path(__file__).resolve().parents[1] / "lean-toolchain")
            .read_text(encoding="utf-8")
            .strip(),
        )
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
