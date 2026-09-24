#!/usr/bin/env python
"""Enumerate finite corpus laws and distinguish their conditioning conventions."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import itertools
from pathlib import Path

from model_rg.provenance import sha256, write_json


def zero():
    return [[F(0), F(0)], [F(0), F(0)]]


def add(a, b):
    return [[x + y for x, y in zip(u, v)] for u, v in zip(a, b)]


def scale(a, c):
    return [[c * x for x in row] for row in a]


def outer(x, y):
    return [[u * v for v in y] for u in x]


def sandwich(a, covariance):
    return [[sum((a[i][k] * covariance[k][l] * a[j][l]
                  for k in range(2) for l in range(2)), F(0))
             for j in range(2)] for i in range(2)]


def matrix_sum(values):
    result = zero()
    for value in values:
        result = add(result, value)
    return result


def moments(law):
    if sum((p for p, _ in law), F(0)) != 1:
        raise AssertionError('The enumerated law is not normalized')
    mean = [sum((p * x[j] for p, x in law), F(0)) for j in range(2)]
    covariance = matrix_sum([scale(outer([x[j] - mean[j] for j in range(2)],
                                        [x[j] - mean[j] for j in range(2)]), p)
                             for p, x in law])
    return mean, covariance


def iid_corpora(support, size):
    output = []
    for indices in itertools.product(range(len(support)), repeat=size):
        probability = F(1)
        corpus = []
        for index in indices:
            p, x = support[index]
            probability *= p
            corpus.append(x)
        output.append((probability, corpus))
    return output


def clustered_corpora():
    output = []
    # Two independent documents, each with a shared sign and two independent
    # block signs. The latter have amplitude 1/2 in the emitted vectors.
    for signs in itertools.product((-1, 1), repeat=6):
        corpus = []
        for document in range(2):
            common = F(signs[document])
            for block in range(2):
                noise = F(signs[2 + 2 * document + block], 2)
                corpus.append((common + noise, 2 * common - noise))
        output.append((F(1, 64), corpus))
    return output


def transports(count, mode):
    if mode == 'matrix':
        return [[[F(1, k + 1), F((-1) ** k, 3)], [F(k, 5), F(2, 3)]]
                for k in range(count)]
    weights = [F((-1) ** k) if mode == 'alternating' else F(1)
               for k in range(count)]
    return [[[w, F(0)], [F(0), w]] for w in weights]


def covariance_case(name, law, batch, count, mode, iid_covariance=None,
                    cluster_covariance=None):
    size = len(law[0][1])
    if batch * count > size:
        raise AssertionError('An enumerated trajectory exceeds its corpus')
    matrices = transports(count, mode)
    total = matrix_sum(matrices)
    corpus_means = []
    average_population_covariance = zero()
    emitted = []
    orderings = list(itertools.permutations(range(size)))
    for probability, corpus in law:
        mean, covariance = moments([(F(1, size), x) for x in corpus])
        corpus_means.append((probability, mean))
        average_population_covariance = add(average_population_covariance,
                                            scale(covariance, probability))
        for ordering in orderings:
            response = [F(0), F(0)]
            for t, matrix in enumerate(matrices):
                block = ordering[t * batch:(t + 1) * batch]
                value = [sum((corpus[i][j] for i in block), F(0)) / batch
                         for j in range(2)]
                response = [response[j] + sum((matrix[j][k] * value[k]
                                               for k in range(2)), F(0))
                            for j in range(2)]
            emitted.append((probability / len(orderings), response))
    _, mean_covariance = moments(corpus_means)
    _, measured = moments(emitted)
    first = scale(matrix_sum([sandwich(a, average_population_covariance)
                              for a in matrices]), F(size, batch * (size - 1)))
    subtraction = scale(sandwich(total, average_population_covariance), F(-1, size - 1))
    conditional = add(first, subtraction)
    corpus_term = sandwich(total, mean_covariance)
    if measured != add(conditional, corpus_term):
        raise AssertionError('Total covariance differs from complete corpus/order enumeration')
    if iid_covariance is not None:
        independent = scale(matrix_sum([sandwich(a, iid_covariance) for a in matrices]), F(1, batch))
        if measured != independent:
            raise AssertionError('The independent-corpus cancellation failed')
    if cluster_covariance is not None:
        marginal, distinct = cluster_covariance
        contrast = add(marginal, scale(distinct, F(-1)))
        prediction = add(scale(matrix_sum([sandwich(a, contrast) for a in matrices]), F(1, batch)),
                         sandwich(total, distinct))
        if measured != prediction:
            raise AssertionError('The retained document covariance differs')
    return dict(name=name, population=size, batch=batch, batches=count, transport=mode,
                corpus_outcomes=len(law), orderings_per_corpus=len(orderings),
                covariance=[[str(x) for x in row] for row in measured],
                mean_conditional_covariance=[[str(x) for x in row] for row in conditional],
                corpus_mean_contribution=[[str(x) for x in row] for row in corpus_term])


def adaptive_path(values):
    theta, memory = F(1, 3), F(0)
    output = []
    for value in values:
        gradient = (theta - value) * (1 + theta * theta)
        clipped = max(F(-1), min(F(1), gradient))
        memory = (memory + clipped) / 2
        theta -= memory / 5
        output.append((theta, memory))
    return tuple(output)


def adaptive_law(length, replacement=False):
    law = defaultdict(F)
    choices = list(itertools.product(range(3), repeat=length) if replacement
                   else itertools.permutations(range(3), length))
    for corpus in itertools.product((-1, 1), repeat=3):
        for indices in choices:
            law[adaptive_path([F(corpus[i]) for i in indices])] += F(1, 8 * len(choices))
    return dict(law)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='docs/corpus-conditioning')
    args = p.parse_args()
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=False)
    scalar = [(F(1, 2), (F(-1), F(-2))), (F(1, 2), (F(1), F(2)))]
    vector = [(F(1, 2), (F(-1), F(0))), (F(1, 3), (F(2), F(1))),
              (F(1, 6), (F(0), F(-2)))]
    records = []
    for batch, count, mode in [(1, 1, 'constant'), (1, 2, 'matrix'),
                                (1, 4, 'constant'), (2, 2, 'alternating')]:
        records.append(covariance_case('independent-scalar', iid_corpora(scalar, 4),
                                       batch, count, mode, iid_covariance=moments(scalar)[1]))
    for count in (1, 2, 3):
        records.append(covariance_case('independent-vector', iid_corpora(vector, 3),
                                       1, count, 'matrix', iid_covariance=moments(vector)[1]))
    cluster_marginal = [[F(5, 4), F(7, 4)], [F(7, 4), F(17, 4)]]
    cluster_distinct = scale([[F(1), F(2)], [F(2), F(4)]], F(1, 3))
    for batch, count, mode in [(1, 2, 'constant'), (1, 4, 'constant'),
                                (1, 4, 'alternating'), (2, 2, 'matrix')]:
        records.append(covariance_case('document-clustered', clustered_corpora(),
                                       batch, count, mode,
                                       cluster_covariance=(cluster_marginal, cluster_distinct)))
    unequal = [(F(1, 3), [(F(i - 2), F(2 * i + 1)) for i in range(3)]),
               (F(2, 3), [(F(3 * i + 1), F(-i - 2)) for i in range(3)])]
    records.append(covariance_case('unequal-correlated', unequal, 1, 3, 'matrix'))
    paths = []
    for length in (1, 2, 3):
        independent = defaultdict(F)
        for sequence in itertools.product((-1, 1), repeat=length):
            independent[adaptive_path([F(x) for x in sequence])] += F(1, 2 ** length)
        actual = adaptive_law(length)
        if actual != dict(independent):
            raise AssertionError('Averaged single-pass and fresh-stream adaptive path laws differ')
        paths.append(dict(updates=length, distinct_native_like_paths=len(actual), exact_equality=True))
    if records[2]['mean_conditional_covariance'] != [['0', '0'], ['0', '0']]:
        raise AssertionError('The fixed-corpus complete-sum variance should vanish')
    if records[2]['covariance'] == records[2]['mean_conditional_covariance']:
        raise AssertionError('Omitting corpus-mean uncertainty was not detected')
    clustered_sum = records[8]['covariance']
    iid_sum = [[str(x) for x in row] for row in scale(cluster_marginal, F(4))]
    if clustered_sum == iid_sum:
        raise AssertionError('Discarding document dependence was not detected')
    if adaptive_law(3, replacement=True) == adaptive_law(3):
        raise AssertionError('Corpus resampling was incorrectly equated with fresh data')
    if (len(records), len(paths)) != (12, 3):
        raise AssertionError('The selected exact-case inventory changed')
    write_json(target/'verification.json', dict(status='passed', exact_cases=15,
        covariance_cases=records, adaptive_path_cases=paths, distinctions_detected=3,
        checker_sha256=sha256(__file__),
        helper_sha256=sha256(Path(__file__).resolve().parents[1]/'src/model_rg/provenance.py'),
        arithmetic='Exact rational enumeration of every selected corpus and ordering outcome.',
        scope='Finite examples check the stand-alone corpus-conditioning identities and adaptive '
              'path-law statement. The recurrence is an exact rational illustration, not native '
              'PLDR training. No new model trajectory, training seed or critical exponent is measured.'))
    print('Corpus-conditioning checks passed: 12 covariance cases, 3 adaptive path cases, 3 distinctions', flush=True)


if __name__ == '__main__':
    main()
