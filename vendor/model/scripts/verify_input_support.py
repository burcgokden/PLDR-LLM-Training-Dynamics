#!/usr/bin/env python
"""Check the unused-input-row condition on the full declared finite corpus."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    study = Path(args.study).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol_path = study / 'protocol.json'
    spec = json.loads(protocol_path.read_text())
    if spec.get('schema') != 'native-invisible-sector-v1':
        raise ValueError('Expected the frozen native support-null experiment')
    token = spec['token']
    corpus = study.parents[1] / 'outer-transfer-20260911/data/corpus-0.npy'
    panels = corpus.parent / 'panels.npz'
    checked = {str(protocol_path): sha256(protocol_path)}
    for path in [corpus, panels]:
        digest = sha256(path)
        if digest != spec['inputs'][str(path)]:
            raise ValueError('Changed input: ' + str(path))
        checked[str(path)] = digest
    data = np.load(corpus, mmap_mode='r', allow_pickle=False)
    if data.shape != (65536, 513):
        raise ValueError('Unexpected document/block layout')
    # Each document supplies eight 64-input/one-external-target blocks.
    # Their input positions are exactly 0,...,511; position 512 is a target.
    corpus_count = sum(int(np.count_nonzero(data[start:start + 1024, :512] == token))
                       for start in range(0, len(data), 1024))
    with np.load(panels, allow_pickle=False) as archive:
        evaluation = archive['evaluation'][:, :64]
        evaluation_count = int(np.count_nonzero(evaluation == token))
        evaluation_positions = int(evaluation.size)
    if corpus_count or evaluation_count:
        raise ValueError('The intervened input row occurs in the declared support')
    write_json(output, dict(
        status='passed', schema='finite-input-support-verification-v1', token=token,
        documents=len(data), supervised_blocks=len(data) * 8,
        corpus_input_positions=len(data) * 512, corpus_occurrences=corpus_count,
        evaluation_input_positions=evaluation_positions,
        evaluation_occurrences=evaluation_count, checked_sha256=checked,
        verifier_sha256=sha256(__file__),
        scope='Every input position in the materialized corpus-zero source blocks and fixed evaluation prefixes. This does not establish absence in the RefinedWeb population or in other prompts.'))
    print('Passed full finite-corpus input-support check')


if __name__ == '__main__':
    main()
