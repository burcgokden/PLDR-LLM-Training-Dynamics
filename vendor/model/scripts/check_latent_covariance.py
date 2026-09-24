"""Exact finite checks for the document-latent correction; no native execution."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time


def variance(values):
    mean = sum(values, F(0)) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.perf_counter()
    rows = []
    for b in range(1, 7):
        signs = list(product((-1, 1), repeat=b + 1))
        for scales in ([F(1)] * b, [F(i + 1) for i in range(b)]):
            sums = [F(b * s[0]) + sum(a * t for a, t in zip(scales, s[1:]))
                    for s in signs]
            actual = variance(sums) / b
            predicted = F(b) + sum(a * a for a in scales) / b
            if actual != predicted:
                raise AssertionError('Heterogeneous residual identity failed')
            rows.append(dict(b=b,scales=[str(x) for x in scales],outcomes=len(signs),
                             chi_exact=str(actual),chi_formula=str(predicted)))
    # A finite two-dimensional conditional model with document-dependent means,
    # position-dependent covariances and conditionally correlated residuals.
    # Enumerate all 2^5 equally weighted outcomes, without Monte Carlo error.
    b = 3
    samples = []
    for doc, common, a, c, d in product((-1, 1), repeat=5):
        eps = (a, c, d)
        y = [[F(doc * (i + 1) + eps[i] + common),
              F(2 * doc + (i + 1) * eps[i] - common)] for i in range(b)]
        samples.append((doc, y))

    def covariance(x):
        n = len(x)
        mean = [sum(row[k] for row in x) / n for k in range(2)]
        return [[sum((row[i] - mean[i]) * (row[j] - mean[j])
                     for row in x) / n for j in range(2)] for i in range(2)]

    totals = [[sum(y[i][k] for i in range(b)) for k in range(2)] for _, y in samples]
    lhs = [[v / b for v in row] for row in covariance(totals)]
    conditional_means, conditional_covariances = [], []
    for doc in (-1, 1):
        group = [x for (s, _), x in zip(samples, totals) if s == doc]
        conditional_means.append([sum(x[k] for x in group) / len(group) for k in range(2)])
        conditional_covariances.append(covariance(group))
    between = covariance(conditional_means)
    rhs = [[(between[i][j] + sum(c[i][j] for c in conditional_covariances) / 2) / b
            for j in range(2)] for i in range(2)]
    if lhs != rhs:
        raise AssertionError('Conditional matrix covariance decomposition failed')
    # Deliberately invalid simplifications are retained as internal controls.
    unequal = next(r for r in rows if r['b'] == 2 and r['scales'] == ['1', '2'])
    if unequal['chi_exact'] != '9/2':
        raise AssertionError('Unequal-variance control failed')
    record = dict(schema='latent-covariance-v1',status='passed',
                  exact_arithmetic='fractions.Fraction, exhaustive finite laws',
                  scalar_cases=rows,conditional_vector_outcomes=len(samples),
                  conditional_vector_lhs=[[str(v) for v in row] for row in lhs],
                  conditional_vector_rhs=[[str(v) for v in row] for row in rhs],
                  internal_controls=dict(unequal_residual_correct='9/2',
                       unequal_residual_unjustified_shortcut='3',
                       perfectly_correlated_residuals_b2_correct='4',
                       perfectly_correlated_residuals_diagonal_only='3'),
                  python=platform.python_version(),elapsed_seconds=time.perf_counter()-start,
                  training_updates=0,native_forward_calls=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Compute the correlated control as well, rather than just storing constants.
    corr = [F(2 * m + 2 * e) for m, e in product((-1, 1), repeat=2)]
    if variance(corr) / 2 != 4:
        raise AssertionError('Correlated residual control failed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['scalar_cases']},indent=2))


if __name__ == '__main__':
    main()
