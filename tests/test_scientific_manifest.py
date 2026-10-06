"""Negative controls for the lightweight scientific manifest entry point."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scientific_manifest_under_test', ROOT / 'scripts/verify_scientific_manifest.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ScientificManifestTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'program.py').write_text('print(1)\n')
        self.refresh()

    def refresh(self):
        files = verifier.payload_files(self.root)
        (self.root / verifier.MANIFEST).write_text(json.dumps({'files': files, 'payload_sha256': verifier.canonical(files)}))

    def test_complete_payload_and_ignored_build_products(self):
        (self.root / 'validation').mkdir()
        (self.root / 'validation/run.json').write_text('{}')
        self.assertEqual(verifier.verify(self.root)['files'], 1)

    def test_added_and_removed_source_require_refresh(self):
        (self.root / 'new.py').write_text('print(2)\n')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            verifier.verify(self.root)
        self.refresh()
        self.assertEqual(verifier.verify(self.root)['files'], 2)
        (self.root / 'program.py').unlink()
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            verifier.verify(self.root)

    def test_false_aggregate_digest_is_rejected(self):
        path = self.root / verifier.MANIFEST
        value = json.loads(path.read_text())
        value['payload_sha256'] = '0' * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'payload digest differs'):
            verifier.verify(self.root)

    def test_modified_bytes_are_rejected(self):
        (self.root / 'program.py').write_text('print(3)\n')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            verifier.verify(self.root)

    def test_unsafe_manifest_path_is_rejected(self):
        for name in ('../outside', '/outside', 'a/../outside', 'a//outside', 'a\\outside'):
            with self.subTest(name=name):
                files = {name: '0' * 64}
                (self.root / verifier.MANIFEST).write_text(json.dumps({'files': files, 'payload_sha256': verifier.canonical(files)}))
                with self.assertRaisesRegex(ValueError, 'Unsafe manifest path'):
                    verifier.verify(self.root)

    def test_file_and_directory_symlinks_are_rejected(self):
        for target, alias in ((self.root / 'program.py', self.root / 'alias.py'), (self.root, self.root / 'alias')):
            with self.subTest(alias=alias.name):
                alias.symlink_to(target)
                with self.assertRaisesRegex(ValueError, 'Symlink'):
                    verifier.verify(self.root)
                alias.unlink()

    def test_manifest_symlink_is_rejected(self):
        original = self.root / verifier.MANIFEST
        saved = self.root / 'saved.json'
        original.rename(saved)
        original.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            verifier.verify(self.root)
