"""Publication links must resolve in the pinned PDF, beyond JSON agreement."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_formal_manifest import validate_claims, validate_statements
from check_published_pdf import verify_pdf


class PublishedPdfTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.pdf = Path(temp.name) / 'publication.pdf'
        self.write_pdf('theorem.2.1', 1)
        self.manifest = {
            'published_pdf': {'arxiv_id': '2609.34130v1',
                              'url': 'https://arxiv.org/pdf/2609.34130v1',
                              'sha256': self.digest(), 'pages': 2,
                              'page_numbering': 'one-based PDF page index'},
            'labels': {'prop:one': {'anchor': 'proposition.2.1', 'number': '2.1'},
                       'eq:inner': {'anchor': 'equation.2.2', 'number': '2.2'}},
            'statements': [{'primary_label': 'prop:one', 'kind': 'proposition',
                            'number': '2.1', 'anchor': 'proposition.2.1', 'counter': 'theorem',
                            'published_anchor': 'theorem.2.1', 'published_pdf_page': 2,
                            'labels': ['prop:one', 'eq:inner'],
                            'statement_labels': ['prop:one'], 'nested_labels': ['eq:inner'],
                            'label_ownership': [{'label': 'prop:one', 'environments': []},
                                                {'label': 'eq:inner', 'environments': ['equation']}]}]}
        self.coverage = {'published_pdf': copy.deepcopy(self.manifest['published_pdf']),
                         'claims': [dict(self.manifest['statements'][0], id='prop:one')]}

    def write_pdf(self, anchor, page):
        writer = PdfWriter()
        for _ in range(2):
            writer.add_blank_page(width=72, height=72)
        writer.add_named_destination(anchor, page)
        with self.pdf.open('wb') as output:
            writer.write(output)

    def digest(self):
        return hashlib.sha256(self.pdf.read_bytes()).hexdigest()

    def test_original_anchor_can_differ_and_nested_labels_stay_separate(self):
        report = verify_pdf(self.pdf, self.manifest, self.coverage)
        self.assertEqual(report['statements_checked'], 1)
        self.assertEqual(report['original_build_anchors_absent_from_pdf'], 1)
        self.assertEqual(self.manifest['labels']['eq:inner']['anchor'], 'equation.2.2')

    def test_changed_pdf_bytes_rejected_before_parsing(self):
        self.pdf.write_bytes(b'changed PDF')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            verify_pdf(self.pdf, self.manifest)

    def test_matching_code_and_data_are_insufficient_when_destination_is_missing(self):
        self.write_pdf('theorem.2.2', 1)
        self.manifest['published_pdf']['sha256'] = self.digest()
        self.coverage['published_pdf'] = copy.deepcopy(self.manifest['published_pdf'])
        validate_claims(self.manifest, self.coverage)
        with self.assertRaisesRegex(ValueError, 'Missing published PDF destination'):
            verify_pdf(self.pdf, self.manifest, self.coverage)

    def test_existing_destination_on_wrong_page_rejected(self):
        self.write_pdf('theorem.2.1', 0)
        self.manifest['published_pdf']['sha256'] = self.digest()
        with self.assertRaisesRegex(ValueError, 'destination page differs'):
            verify_pdf(self.pdf, self.manifest)

    def test_wrong_pdf_page_count_rejected(self):
        self.manifest['published_pdf']['pages'] = 3
        with self.assertRaisesRegex(ValueError, 'page count differs'):
            verify_pdf(self.pdf, self.manifest)

    def test_artifact_identity_requires_version_digest_and_page_convention(self):
        bad = [('arxiv_id', '2609.34130'), ('url', 'https://arxiv.org/pdf/2609.34130v2'),
               ('sha256', 'not-a-sha256'), ('pages', True), ('page_numbering', 'printed pages')]
        for key, value in bad:
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest['published_pdf'][key] = value
                with self.assertRaises(ValueError):
                    validate_statements(manifest)

    def test_published_mapping_is_required_and_pages_are_integer_indices(self):
        for value in [None, True, 0, 3, 2.0]:
            with self.subTest(value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest['statements'][0]['published_pdf_page'] = value
                with self.assertRaises(ValueError):
                    validate_statements(manifest)
        del self.manifest['statements'][0]['published_anchor']
        with self.assertRaises(ValueError):
            validate_statements(self.manifest)

    def test_destination_must_belong_to_statement_identity(self):
        self.manifest['statements'][0]['published_anchor'] = 'theorem.2.2'
        with self.assertRaisesRegex(ValueError, 'identify its statement'):
            validate_statements(self.manifest)

    def test_code_data_disagreement_rejected(self):
        for kind in ['pdf', 'anchor', 'page', 'identity', 'missing', 'duplicate']:
            with self.subTest(kind=kind):
                coverage = copy.deepcopy(self.coverage)
                if kind == 'pdf':
                    coverage['published_pdf']['sha256'] = '0' * 64
                elif kind == 'anchor':
                    coverage['claims'][0]['published_anchor'] = 'proposition.2.1'
                elif kind == 'page':
                    coverage['claims'][0]['published_pdf_page'] = 1
                elif kind == 'identity':
                    coverage['claims'][0]['id'] = 'different:statement'
                elif kind == 'missing':
                    coverage['claims'] = []
                else:
                    coverage['claims'].append(copy.deepcopy(coverage['claims'][0]))
                with self.assertRaises(ValueError):
                    validate_claims(self.manifest, coverage)


if __name__ == '__main__':
    unittest.main()
