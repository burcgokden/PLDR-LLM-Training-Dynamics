# Monograph subjects and scientific code

| Monograph chapters | Subject | Main code |
| --- | --- | --- |
| 2–7 | Row geometry, finite source work, complete optimizer state and PLGA transfer | `vendor/row/PldrLlmCurvatureSandpile/`, `vendor/row/experiments/`, `PldrTrainingDynamics/RowFraction.lean`, `StochasticTransport.lean`, `LimitCriteria.lean`, `TrajectoryChart.lean` |
| 8–11 | Chronological affine blocking, orbit criteria, closure and crossover | `vendor/rg/RowRGMap/`, `vendor/rg/src/row_rgmap/`, `PldrTrainingDynamics/ScalingBalance.lean`, `ScalarFaceMemory.lean` |
| 12–18 | Complete conditional laws, consuming corpus, finite flux and physical clocks | `vendor/model/ModelRG/`, `vendor/model/src/model_rg/`, `vendor/model/scripts/`, `PldrTrainingDynamics/PathErrorBudget.lean` |
| 19–24 | Predictive reduction, cache risk, memory and adaptation | `vendor/model/scripts/cache_state_contract.py`, `analyze_cache_state_transfer.py`, `verify_cache_state_transfer.py`, `vendor/model/src/model_rg/` |
| 25–29 | Conditional scaling, collective limits, inference visibility and readout | `vendor/model/ModelRG/`, `vendor/model/src/model_rg/`, `vendor/model/scripts/` |
| Part VI, Appendices A–F | Statistical roles, complete outcomes, reproduction, formal correspondence, notation and caption lists | Dataset `coverage.json` and `data-dictionary.json`; `provenance/statement-manifest.json`; `docs/EXECUTION.md` |

The namespaces `row`, `rg` and `model` are stable implementation families, not
separate publication dependencies. `provenance/statement-manifest.json` records
all 263 numbered statements, final numbering, stable labels, selected formal
declarations, explicit hypotheses and remaining written obligations. Mathematical
proofs stand alone in the monograph. See the dataset's coverage index for the
reported displays and claim scopes; nested observations are not independent
training replications.

## Statement identities

`provenance/statement-manifest.json` records each numbered statement's
`primary_label`, printed `number`, original-build `anchor`, and shared `counter`.
`statement_labels` contains its own labels, including declared aliases;
`nested_labels` contains equation and clause labels within it. Definition 25.5
uses `model:def:fluctuation-criticality`; its Equation (25.12) uses
`model:eq:criticality`. Proposition 25.12 has its own statement identity.
Equation labels remain valid targets of partial formal correspondence.

Run `python3 scripts/check_formal_manifest.py --data-repo /path/to/data`
to check the code manifest and numerical dataset's mathematical claim index
together. Neither manuscript sources nor TeX are required.

## Published PDF destinations

The original-build `anchor` and `labels` records remain unchanged. Statement
rows additionally expose `published_anchor` and `published_pdf_page` for the
published arXiv monograph. `published_pdf_page` is a one-based PDF page index,
including front matter; it is distinct from printed page numbers.

The top-level `published_pdf` binding identifies arXiv `2609.34130v1`, its
versioned PDF URL, SHA-256 and 655-page extent. All 263 numbered statements
have published destinations. The 212 original-build anchors that differ
remain available as provenance. Nested equation/clause anchors and the
original label map retain their original-build interpretation.

`check_formal_manifest.py --data-repo DATA-REPO` checks both original metadata
and the published-PDF mapping for code/data agreement. To authenticate the
actual PDF and resolve every published statement destination and page, run
from this code package's root (the included `companions/dynamics/` directory
when using an included copy):

```sh
python3 -m pip install -r requirements-publication.txt
curl -fL https://arxiv.org/pdf/2609.34130v1 -o /tmp/pldr-monograph-2609.34130v1.pdf
python3 scripts/check_published_pdf.py --pdf /tmp/pldr-monograph-2609.34130v1.pdf --data-repo DATA-REPO --output validation/published-pdf.json
```

Use matching code and dataset revisions that include these fields. The PDF
checker requires only the optional pinned `pypdf` dependency in addition to
the standard library; it reads a local PDF and never changes the dataset.
The ordinary evidence reader and code/data metadata checker remain
standard-library-only. The report distinguishes actual PDF verification from
metadata agreement. This check covers numbered-statement navigation, with
mathematical proofs and numerical reproduction retaining their separate checks.

The two-row example in Chapter 6 is illustrative mathematics. Run
`python3 scripts/check_chart_example.py` to check its exact rational values
and the fixed/adapted-slice counterexample. Its sensitivities are conditional
hypotheses and do not estimate a trained model.
