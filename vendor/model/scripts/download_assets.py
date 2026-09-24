#!/usr/bin/env python
"""Fetch immutable public checkpoints and verify a shared tokenizer."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from model_rg.provenance import sha256, write_json

MODELS={
    'PLDR-LLM-v51-SOC-110M-1':'7a34e2ca9aa78038683677cfda17fe3a9fe6da8a',
    'PLDR-LLM-v51-SOC-110M-2':'5e6f3164f6aef7878743f92725364860931047a0',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);records={}
    for name,revision in MODELS.items():
        path=Path(snapshot_download('fromthesky/'+name,revision=revision,local_dir=root/name,
             allow_patterns=['*.py','*.json','*.model','*.safetensors','README.md','requirements.txt'],max_workers=4))
        records[name]={'repository':'fromthesky/'+name,'revision':revision,
                       'files':{str(f.relative_to(path)):sha256(f) for f in sorted(path.iterdir()) if f.is_file()}}
    hashes={record['files']['tokenizer.model'] for record in records.values()}
    if len(hashes)!=1:raise RuntimeError('Checkpoints use different tokenizers')
    write_json(root/'pinned-assets.json',records)


if __name__=='__main__':main()
