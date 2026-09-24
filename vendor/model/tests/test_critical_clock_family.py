"""Resource-law checks for matched nominal single-pass clocks."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_critical_clock_family import design_at
import prepare_critical_refinement as refinement
from verify_critical_onepass import source_population


def member(n):
    spec = dict(name='fixture', heads=[2, 4, 8, 14, 24], rate_ratios=[0., 2., 4.],
                seeds=[11, 12, 13], environments=[0, 1], steps_per_head=256,
                corpus_documents_per_head=8192, shared_seed=7)
    design = design_at(spec, n)
    return dict(design=design, source=dict(partition_seed=9152501,
        population_policy='nested-width-prefix-v1', population_documents=8192*n,
        population_blocks=65536*n), joint_family=dict(specification=spec, consumed_fraction=.125))


def test_nominal_clocks_and_source_fraction_match_at_every_width():
    for n in [2, 4, 8, 14, 24]:
        p = member(n)
        d = p['design']
        assert d['steps']*32/p['source']['population_blocks'] == .125
        assert .0006*d['steps']/n == pytest.approx(.1536)
        np.testing.assert_allclose(np.array(d['controls'])*.0003*d['steps'], [0., .3072, .6144])


def test_populations_are_nested_and_disjoint_without_changing_the_master():
    p4 = source_population(member(4), 0)
    p14 = source_population(member(14), 0)
    other = source_population(member(14), 1)
    np.testing.assert_array_equal(p4, p14[:len(p4)])
    assert len(np.unique(p14)) == 8192*14
    assert not np.intersect1d(p14, other).size
    assert np.all((p14 >= 0) & (p14 < 524288))


def test_fixed_resource_substitution_or_wrong_rate_ratio_is_rejected():
    original = member(4)
    for location, key, value in [('source', 'population_blocks', 2097152),
                                  ('design', 'steps', 2048),
                                  ('design', 'controls', [0., 2., 4.])]:
        p = copy.deepcopy(original)
        p[location][key] = value
        with pytest.raises(ValueError):
            source_population(p, 0)
    p = copy.deepcopy(original)
    p['source']['population_policy'] = 'undeclared'
    with pytest.raises(ValueError):
        source_population(p, 0)


def test_full_half_source_reconstruction_is_unchanged():
    p = dict(source=dict(partition_seed=9152501, population_documents=262144,
                        population_blocks=2097152), design=dict(steps=2048))
    expected = np.random.default_rng(9152501).permutation(524288).reshape(2, -1)
    for e in [0, 1]:
        np.testing.assert_array_equal(source_population(p, e), expected[e])


def test_refinement_cannot_reuse_search_initializations(tmp_path, monkeypatch):
    monkeypatch.setattr(refinement, 'ROOT', tmp_path)
    search = tmp_path/'search.json'
    search.write_text(json.dumps(dict(status='complete', design=dict(seeds=[11, 12, 13]))))
    decision = tmp_path/'decision.json'
    decision.write_text(json.dumps(dict(search_analyses={str(search): refinement.sha256(search)},
        hypotheses=['fixture'], primary_fields=['row'], held_out_axes=['seed'], decision_rules=['fixture'])))
    design = tmp_path/'design.json'
    design.write_text(json.dumps(dict(seeds=[13, 21, 22])))
    called = []
    monkeypatch.setattr(refinement, 'prepare', lambda *args: called.append(args))
    with pytest.raises(ValueError, match='independent initialization seeds'):
        refinement.prepare_refinement(tmp_path/'new-study', design, decision)
    assert called == []
    assert not (tmp_path/'new-study').exists()
