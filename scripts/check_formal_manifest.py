#!/usr/bin/env python3
"""Check label/module correspondence without manuscript sources.

Type elaboration and axiom checks are performed by check_lean.py. This check
binds selected scope descriptions to shipped sources, not prose equivalence.
"""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
KINDS = {'theorem', 'proposition', 'lemma', 'corollary', 'definition',
         'assumption', 'prediction'}


def validate_publication(pdf, rows):
    """Check the separate published-PDF binding without importing a PDF parser.

    The legacy anchor/label records continue to identify the original build.
    Actual destination existence is checked by check_published_pdf.py.
    """
    if not isinstance(pdf, dict):
        raise ValueError('Missing published PDF identity')
    arxiv_id = pdf.get('arxiv_id', '')
    if not isinstance(arxiv_id, str) or not re.fullmatch(r'\d{4}\.\d{4,5}v[1-9]\d*', arxiv_id):
        raise ValueError('Published PDF needs a versioned arXiv identifier')
    if pdf.get('url') != 'https://arxiv.org/pdf/' + arxiv_id:
        raise ValueError('Published PDF URL/version differs')
    sha = pdf.get('sha256', '')
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
        raise ValueError('Invalid published PDF SHA-256')
    pages = pdf.get('pages')
    if type(pages) is not int or pages < 1:
        raise ValueError('Invalid published PDF page count')
    if pdf.get('page_numbering') != 'one-based PDF page index':
        raise ValueError('Unspecified published PDF page convention')
    anchors = set()
    for row in rows:
        anchor = row.get('published_anchor')
        if not isinstance(anchor, str) or anchor not in {
                row['kind'] + '.' + row['number'], row['counter'] + '.' + row['number']}:
            raise ValueError('Published anchor does not identify its statement')
        if anchor in anchors:
            raise ValueError('Duplicate published statement anchor')
        anchors.add(anchor)
        page = row.get('published_pdf_page')
        if type(page) is not int or not 1 <= page <= pages:
            raise ValueError('Invalid published statement PDF page')


def validate_statements(manifest):
    """Check exported ownership, counter identities and uniqueness, without TeX."""
    labels = manifest['labels']
    primaries, anchors = set(), set()
    for row in manifest['statements']:
        primary, kind = row['primary_label'], row['kind']
        number, anchor, counter = row['number'], row['anchor'], row['counter']
        if kind not in KINDS or counter not in KINDS or not number:
            raise ValueError('Invalid statement kind, counter or number: ' + primary)
        if anchor not in {kind + '.' + number, counter + '.' + number}:
            raise ValueError('Nonstatement primary anchor: ' + primary)
        owners, nested = set(row['statement_labels']), set(row['nested_labels'])
        names = row['labels']
        if (not owners or primary not in owners or owners & nested or
                len(names) != len(set(names)) or set(names) != owners | nested or
                any(name not in labels for name in names)):
            raise ValueError('Inconsistent statement labels: ' + primary)
        ownership = row['label_ownership']
        if (len(ownership) != len(names) or
                {r['label'] for r in ownership} != set(names) or
                {r['label'] for r in ownership if not r['environments']} != owners):
            raise ValueError('Inconsistent owning-label metadata: ' + primary)
        for name in owners:
            if labels[name]['number'] != number or labels[name]['anchor'] != anchor:
                raise ValueError('Stale or ambiguous compiled identity: ' + name)
        if primary in primaries or anchor in anchors:
            raise ValueError('Duplicate primary statement identity: ' + primary)
        primaries.add(primary)
        anchors.add(anchor)
    validate_publication(manifest.get('published_pdf'), manifest['statements'])
    return primaries


def validate_claims(manifest, coverage):
    statements = {row['primary_label']: row for row in manifest['statements']}
    claims = [row for row in coverage['claims'] if row['kind'] in KINDS]
    validate_publication(coverage.get('published_pdf'), claims)
    if coverage['published_pdf'] != manifest['published_pdf']:
        raise ValueError('Code/data published PDF identity differs')
    if len(claims) != len(statements) or {row['id'] for row in claims} != set(statements):
        raise ValueError('Code/data primary statement identities differ')
    for claim in claims:
        row = statements[claim['id']]
        for field in ['kind', 'number', 'anchor', 'counter', 'statement_labels', 'nested_labels',
                      'published_anchor', 'published_pdf_page']:
            if claim[field] != row[field]:
                raise ValueError('Code/data statement ' + field + ' differs: ' + claim['id'])
    return len(claims)


def check(root=ROOT, data=None):
    root=Path(root);m=json.loads((root/'provenance/statement-manifest.json').read_text());labels=m['labels'];names=set()
    validate_statements(m)
    if data is not None:
        validate_claims(m, json.loads((Path(data)/'coverage.json').read_text()))
    for r in m['formal_mappings']:
        if any(l not in labels for l in r['labels']):raise ValueError('Unresolved formal label')
        if not r.get('checked_hypotheses') or not r.get('remaining_written_obligations'):raise ValueError('Missing partial-coverage boundary')
        if r.get('module'):
            p=root/r['module']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['companion_source_sha256']:raise ValueError('Changed formal source '+r['module'])
        names.update(r.get('declarations',[]))
        for c in r.get('clauses') or []:
            names.update(c['declarations'])
            module=c['module']
            if module is None:
                if c['declarations']:raise ValueError('Declarations without a formal module')
                continue
            p=root/'vendor/rg'/Path(*module.split('.')).with_suffix('.lean')
            if not p.is_file():raise FileNotFoundError(p)
    return {'status':'passed','statements':len(m['statements']),'mapped_declarations':len(names),'code_data_identity_agreement': data is not None,
            'published_pdf':m['published_pdf'], 'published_pdf_destinations_checked':False,
            'scope':'Owning statement labels, original-build counters, published-PDF binding and module integrity; actual PDF destinations, exact types and axioms are checked separately.'}
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-repo',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(data=args.data_repo),indent=2))
