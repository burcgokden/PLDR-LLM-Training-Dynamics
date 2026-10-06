"""Check the public dependency paths needed by a fresh Lake checkout."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ('lake-manifest.json', 'vendor/row/lake-manifest.json',
             'vendor/rg/lake-manifest.json', 'vendor/model/lake-manifest.json')


class LeanDependencyManifestTests(unittest.TestCase):
    def test_pinned_packages_have_loadable_configuration_paths(self):
        for name in MANIFESTS:
            manifest = json.loads((ROOT / name).read_text())
            self.assertTrue(manifest['packages'], name)
            for package in manifest['packages']:
                with self.subTest(manifest=name, package=package['name']):
                    self.assertIn(package['configFile'], ('lakefile.lean', 'lakefile.toml'))
                    self.assertEqual(package['manifestFile'], 'lake-manifest.json')
                    self.assertEqual(package['type'], 'git')
                    self.assertTrue(package['url'].startswith('https://github.com/'))
                    self.assertRegex(package['rev'], r'^[0-9a-f]{40}$')

    def test_vendor_packages_use_the_same_pinned_dependencies(self):
        def dependencies(name):
            manifest = json.loads((ROOT / name).read_text())
            return {package['name']: {key: package[key] for key in
                    ('url', 'rev', 'configFile', 'manifestFile')}
                    for package in manifest['packages']}

        expected = dependencies(MANIFESTS[0])
        for name in MANIFESTS[1:]:
            with self.subTest(manifest=name):
                self.assertEqual(dependencies(name), expected)


if __name__ == '__main__':
    unittest.main()
