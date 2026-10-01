#!/usr/bin/env python3
"""Check statement destinations in the exact, separately supplied arXiv PDF.

The PDF is read locally; this command never downloads or changes evidence.
Only this publication check needs pypdf. Evidence readers remain stdlib-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

from check_formal_manifest import ROOT, check as check_formal
from check_formal_manifest import validate_claims, validate_statements


def verify_pdf(path, manifest, coverage=None):
    validate_statements(manifest)
    if coverage is not None:
        validate_claims(manifest, coverage)
    binding = manifest['published_pdf']
    with Path(path).open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
        if sha != binding['sha256']:
            raise ValueError('Published PDF SHA-256 differs from the pinned artifact')
        source.seek(0)
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError('Install requirements-publication.txt for the PDF check') from exc
        reader = PdfReader(source)
        if len(reader.pages) != binding['pages']:
            raise ValueError('Published PDF page count differs')
        destinations = reader.named_destinations
        checked = []
        for row in manifest['statements']:
            anchor = row['published_anchor']
            if anchor not in destinations:
                raise ValueError('Missing published PDF destination: ' + anchor)
            page_index = reader.get_destination_page_number(destinations[anchor])
            if page_index is None or page_index < 0:
                raise ValueError('Unresolved published PDF destination page: ' + anchor)
            page = page_index + 1
            if page != row['published_pdf_page']:
                raise ValueError('Published PDF destination page differs: ' + anchor)
            checked.append({'id': row['primary_label'], 'number': row['number'],
                            'original_build_anchor': row['anchor'],
                            'published_anchor': anchor, 'published_pdf_page': page})
        original_missing = sum(row['anchor'] not in destinations for row in manifest['statements'])
    return {'status': 'passed', 'published_pdf': binding,
            'statements_checked': len(checked), 'destinations_in_pdf': len(destinations),
            'original_build_anchors_absent_from_pdf': original_missing,
            'code_data_identity_agreement': coverage is not None, 'statements': checked,
            'scope': 'Pinned PDF bytes and numbered-statement destinations/pages; no proof or numerical replication claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--data-repo', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check_formal(data=args.data_repo)
    manifest = json.loads((ROOT / 'provenance/statement-manifest.json').read_text())
    coverage = None if args.data_repo is None else json.loads((args.data_repo / 'coverage.json').read_text())
    report = verify_pdf(args.pdf, manifest, coverage)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
