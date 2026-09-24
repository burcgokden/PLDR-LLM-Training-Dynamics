"""Rebuild all owned Lean modules without any pre-existing owned build outputs."""
from companion_paths import legacy_path
import argparse,os,shutil,subprocess,tempfile,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json

def main(a):
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ELAN_HOME=legacy_path('/pldr-tools/elan'),PATH=legacy_path('/pldr-tools/elan/bin')+os.pathsep+os.environ['PATH'])
    with tempfile.TemporaryDirectory(prefix='modelrg-clean-',dir='/tmp') as tmp:
        root=Path(tmp);shutil.copytree(REPO/'ModelRG',root/'ModelRG')
        for n in ['ModelRG.lean','lakefile.toml','lake-manifest.json','lean-toolchain']:shutil.copy2(REPO/n,root/n)
        (root/'.lake').mkdir();(root/'.lake/packages').symlink_to(REPO/'.lake/packages',target_is_directory=True)
        with (out/'build.log').open('w') as f:subprocess.run(['lake','build'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        with (out/'axioms.log').open('w') as f:subprocess.run(['lake','env','lean',str(REPO/'scripts/formal/Gate.lean')],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write_json(out/'verification.json',dict(status='passed',owned_modules=len(list((REPO/'ModelRG').rglob('*.lean'))),dependency_artifacts_reused=True,preexisting_owned_artifacts=False,checker_sha256=sha256(__file__),module_sources={str(p.relative_to(REPO)):sha256(p) for p in (REPO/'ModelRG').glob('*.lean')},build_sha256=sha256(out/'build.log'),gate_sha256=sha256(out/'axioms.log')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
