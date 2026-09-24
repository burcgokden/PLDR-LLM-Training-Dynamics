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
`primary_label`, printed `number`, compiled `anchor`, and shared `counter`.
`statement_labels` contains its own labels, including declared aliases;
`nested_labels` contains equation and clause labels within it. Definition 25.5
uses `model:def:fluctuation-criticality`; its Equation (25.12) uses
`model:eq:criticality`. Proposition 25.12 has its own statement identity.
Equation labels remain valid targets of partial formal correspondence.

Run `python3 scripts/check_formal_manifest.py --data-repo /path/to/data`
to check the code manifest and numerical dataset's mathematical claim index
together. Neither manuscript sources nor TeX are required.

The two-row example in Chapter 6 is illustrative mathematics. Run
`python3 scripts/check_chart_example.py` to check its exact rational values
and the fixed/adapted-slice counterexample. Its sensitivities are conditional
hypotheses and do not estimate a trained model.
