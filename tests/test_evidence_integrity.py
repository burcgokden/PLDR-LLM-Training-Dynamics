"""Negative controls for portable evidence authentication and extraction."""
import gzip,hashlib,json
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evidence_store import Evidence,safe,digest

class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'data';self.root.mkdir()
        raw=b'{"signed_work": -2.5, "passed": false}\n';z=gzip.compress(raw,mtime=0);h=digest(z);self.obj='objects/'+h[:2]+'/'+h+'.gz';p=self.root/self.obj;p.parent.mkdir(parents=True);p.write_bytes(z)
        self.row=dict(id='row/result.json',object=self.obj,sha256=digest(raw),bytes=len(raw),compressed_sha256=h,compressed_bytes=len(z))
        self.index={'schema':'pldr-numerical-evidence-v2','records':[self.row]};self.coverage={'displays':[{'id':'table-1','records':['row/result.json'],'disposition':'complete signed control'}],'claims':[]};self.save()
    def save(self):
        for name,v in [('index.json',self.index),('coverage.json',self.coverage)]:self.root.joinpath(name).write_text(json.dumps(v))
        files={str(p.relative_to(self.root)):digest(p.read_bytes()) for p in self.root.rglob('*') if p.is_file() and p.name!='manifest.json'}
        self.root.joinpath('manifest.json').write_text(json.dumps({'files':files}))
    def test_signed_failure_control_survives_extraction(self):
        target=Path(self.temp.name)/'extracted';r=Evidence(self.root).verify(target)
        self.assertEqual(r['status'],'passed');self.assertEqual(json.loads((target/'row/result.json').read_text()),{'signed_work':-2.5,'passed':False})
    def test_changed_compressed_bytes_rejected(self):
        (self.root/self.obj).write_bytes(b'changed')
        with self.assertRaises(ValueError):Evidence(self.root).verify()
    def test_lfs_pointer_explains_materialization_for_read_and_verify(self):
        for newline in [b'\n',b'\r\n']:
            pointer=newline.join([b'version https://git-lfs.github.com/spec/v1',b'oid sha256:'+self.row['compressed_sha256'].encode(),b'size '+str(self.row['compressed_bytes']).encode(),b''])
            (self.root/self.obj).write_bytes(pointer)
            for action in [lambda: Evidence(self.root).read('row/result.json'),lambda: Evidence(self.root).verify()]:
                with self.subTest(newline=newline,action=action),self.assertRaisesRegex(ValueError,'Unresolved Git LFS pointer: .*git lfs pull origin'):
                    action()
    def test_cache_metadata_is_rejected_with_layout_guidance(self):
        cache=self.root/'.cache/huggingface/download/example.metadata';cache.parent.mkdir(parents=True);cache.write_text('cache')
        with self.assertRaisesRegex(ValueError,r'Dataset file inventory differs.*unexpected=.*\.cache/huggingface/download/example.metadata.*outside its root'):
            Evidence(self.root).verify()
    def test_missing_object_is_named(self):
        (self.root/self.obj).unlink()
        with self.assertRaisesRegex(ValueError,'Dataset file inventory differs.*missing=.*'+self.obj):
            Evidence(self.root).verify()
    def test_index_cannot_be_silently_changed(self):
        self.row['bytes']+=1;self.root.joinpath('index.json').write_text(json.dumps(self.index))
        with self.assertRaises(ValueError):Evidence(self.root).verify()
    def test_duplicate_logical_identity_rejected(self):
        self.index['records'].append(dict(self.row));self.save()
        with self.assertRaisesRegex(ValueError,'Duplicate'):Evidence(self.root)
    def test_unresolved_display_rejected_even_with_valid_file_hashes(self):
        self.coverage['displays'][0]['records']=['missing.json'];self.save()
        with self.assertRaisesRegex(ValueError,'Unresolved'):Evidence(self.root).verify()
    def test_parent_path_extraction_rejected(self):
        for path in ['../outside.json','/absolute.json','a/../../outside','a\\outside']:
            with self.subTest(path=path), self.assertRaises(ValueError):safe(self.root,path)
    def test_symlink_component_rejected(self):
        outside=Path(self.temp.name)/'outside';outside.mkdir();(self.root/'alias').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlink'):safe(self.root,'alias/value.json')
    def test_existing_different_extraction_is_not_overwritten(self):
        target=Path(self.temp.name)/'extracted';p=target/'row/result.json';p.parent.mkdir(parents=True);p.write_text('user data')
        with self.assertRaises(FileExistsError):Evidence(self.root).verify(target)
        self.assertEqual(p.read_text(),'user data')
if __name__=='__main__':unittest.main()
