"""Geometric checks for the bounded magnetic invariant used by native readouts."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from physical_collective_validation import simplex_projection


def test_simplex_projection_color_symmetry_and_fixed_points():
    rng=np.random.default_rng(771)
    for q in [2,3]:
        x=rng.normal(size=(100,q));permutation=rng.permutation(q)
        p=simplex_projection(x)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
        assert np.min(p)>=0
        np.testing.assert_allclose(simplex_projection(x[:,permutation]),p[:,permutation],atol=1e-14)
        np.testing.assert_allclose(simplex_projection(p),p,atol=1e-14)


def test_simplex_projection_variational_characterization():
    rng=np.random.default_rng(772)
    for q in [2,3]:
        x=rng.normal(size=(150,q))*3
        p=simplex_projection(x)
        # The normal-cone inequality characterizes the Euclidean projection
        # onto the entire simplex; it suffices to check every simplex vertex.
        for vertex in np.eye(q):
            assert np.max(np.sum((x-p)*(vertex-p),axis=1))<1e-12


def test_magnetic_invariant_bound_at_simplex_vertices_and_interior():
    rng=np.random.default_rng(773)
    for q in [2,3]:
        truth=np.concatenate([np.eye(q),rng.dirichlet(np.ones(q),size=300)])
        estimate=simplex_projection(rng.normal(size=truth.shape))
        v=np.sqrt(q/(q-1))*(truth-1/q)
        w=np.sqrt(q/(q-1))*(estimate-1/q)
        error=np.abs(np.square(v).sum(1)-np.square(w).sum(1))
        assert np.all(error<=2*np.linalg.norm(v-w,axis=1)+1e-13)
        assert np.max(np.square(w).sum(1))<=1+1e-13
