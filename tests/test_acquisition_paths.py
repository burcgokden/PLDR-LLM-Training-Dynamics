"""Negative controls for exact acquisition identities and immutable parents."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/model/scripts'))
sys.path.insert(0,str(ROOT/'vendor/model/src'))
from acquisition_paths import HISTORICAL_ROOT, resolve_recorded
from cache_state_contract import validate_protocol

class AcquisitionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.asset=self.base/'protocol.json';self.asset.write_bytes(b'expected acquisition bytes')
        self.identity=str(HISTORICAL_ROOT/'example/protocol.json')
        self.digest=hashlib.sha256(self.asset.read_bytes()).hexdigest()
        self.data=dict(schema='pldr-acquisition-locations-v1',logical_root=str(HISTORICAL_ROOT),files=[dict(identity=self.identity,path=str(self.asset),sha256=self.digest)])
        self.manifest=self.base/'locations.json';self.save()
        self.env=patch.dict(os.environ,PLDR_INPUT_MANIFEST=str(self.manifest));self.env.start();self.addCleanup(self.env.stop)
    def save(self):self.manifest.write_text(json.dumps(self.data))
    def test_exact_alias_accepts_recorded_digest(self):
        self.assertEqual(resolve_recorded(self.identity,self.digest),self.asset)
    def test_explicit_logical_root_accepts_immutable_foreign_identity(self):
        self.data['logical_root']='/acquisition/model'
        self.data['files'][0]['identity']='/acquisition/model/example/protocol.json'
        self.save()
        self.assertEqual(resolve_recorded('/acquisition/model/example/protocol.json', self.digest), self.asset)
    def test_noncanonical_logical_root_rejected(self):
        self.data['logical_root']='/acquisition/../model'
        self.save()
        with self.assertRaisesRegex(ValueError,'Unsupported'):resolve_recorded(self.identity, self.digest)
    def test_missing_alias_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing'):resolve_recorded(HISTORICAL_ROOT/'unknown/protocol.json',self.digest)
    def test_duplicate_alias_rejected(self):
        self.data['files'].append(copy.deepcopy(self.data['files'][0]));self.save()
        with self.assertRaisesRegex(ValueError,'Ambiguous'):resolve_recorded(self.identity,self.digest)
    def test_changed_bytes_rejected(self):
        self.asset.write_bytes(b'changed acquisition bytes')
        with self.assertRaisesRegex(ValueError,'Changed'):resolve_recorded(self.identity,self.digest)
    def test_same_name_different_asset_rejected(self):
        other=self.base/'elsewhere/protocol.json';other.parent.mkdir();other.write_bytes(b'unrelated')
        self.data['files'][0].update(path=str(other),sha256=hashlib.sha256(other.read_bytes()).hexdigest());self.save()
        with self.assertRaisesRegex(ValueError,'expected acquisition bytes'):resolve_recorded(self.identity,self.digest)
    def test_wrong_parent_rejected_without_reading_assets(self):
        p=json.loads((ROOT/'tests/fixtures/cache-state-protocol.json').read_text());p['parent']=str(HISTORICAL_ROOT/'wrong-parent')
        study=self.base/'study';study.mkdir();(study/'protocol.json').write_text(json.dumps(p))
        with self.assertRaisesRegex(ValueError,'Invalid parent'):validate_protocol(study)

if __name__=='__main__':unittest.main()
