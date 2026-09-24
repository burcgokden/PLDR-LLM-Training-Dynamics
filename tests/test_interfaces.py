"""Cross-description checks, including boundaries where simplification fails."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/row'),str(ROOT/'vendor/rg/src'),
              str(ROOT/'vendor/model/src'),str(ROOT/'vendor/model/scripts')]
from row_rgmap.core import Edge, block_edges, canonical_edge, canonical_edges, normalize
from experiments.confirm.source_resolved_energy import gate_shape_energy, source_work_gram
from analyze_row_path_confirmation import transport

def centered(a): return a-a.mean(axis=-2,keepdims=True)
def energy(a): return float(np.square(a).sum())
def logsoftmax(z):
    shift=z-z.max()
    return shift-np.log(np.exp(shift).sum())

class InterfaceTests(unittest.TestCase):
    def test_gate_shape_energy_matches_absolute_quotient(self):
        rng=np.random.default_rng(371)
        shape=rng.normal(size=(4,6));gate=np.array([0.,1.,-2.,0.,.3,1.2])
        rows=shape*gate+np.arange(6)[None,:]
        result=gate_shape_energy(shape,gate)
        self.assertAlmostEqual(float(result['energy']),energy(centered(rows)),places=12)

    def test_canonical_face_crossing_retains_memory_loss(self):
        path=[1.,0.,2.,8.]
        blocked=block_edges(canonical_edges(path))
        self.assertEqual(blocked,Edge(0.,8.))
        self.assertEqual(blocked.act(91.),8.)
        self.assertNotEqual(blocked,canonical_edge(path[0],path[-1]))

    def test_positive_tail_telescopes_across_blockings(self):
        path=[3.,2.,8.,1.,.5]
        edges=canonical_edges(path)
        self.assertEqual(block_edges(edges),block_edges([block_edges(edges[:2]),block_edges(edges[2:])]))
        self.assertAlmostEqual(block_edges(edges).act(path[0]),path[-1])

    def test_endpoint_gauge_requires_intermediate_cancellation(self):
        edges=[Edge(.2,1.),Edge(.5,3.)]
        changed=[normalize(edges[0],2.,7.),normalize(edges[1],7.,5.)]
        self.assertEqual(block_edges(changed),normalize(block_edges(edges),2.,5.))

    def test_source_cross_terms_can_cancel_charge(self):
        z=np.array([[1.,-1.],[-1.,1.]])
        d=np.array([[2.,0.],[-2.,0.]])
        result=source_work_gram(z,{'a':d,'b':-d})
        self.assertAlmostEqual(float(result['predicted_energy']),energy(z))
        self.assertGreater(energy(d)+energy(-d),0.)

    def test_scalar_and_batched_face_policy(self):
        rows=np.ones((2,2));delta=np.array([[1.,-1.],[-1.,1.]])
        for source,change in [(rows,delta),(rows[None,...],delta[None,...])]:
            result=source_work_gram(source,{'creation':change})
            self.assertFalse(bool(np.any(result['gain_defined'])))
            self.assertTrue(np.all(result['source_gain']==0.))
            self.assertTrue(np.all(result['predicted_energy']>0.))

    def test_absolute_work_and_normalized_flux_match_endpoints(self):
        rng=np.random.default_rng(863)
        for _ in range(32):
            a=rng.normal(size=(4,5));d=.3*rng.normal(size=a.shape);b=a+d
            z=centered(a);dz=centered(d)
            work=-2*np.vdot(z,dz);charge=energy(dz)
            e1=energy(z)-work+charge
            self.assertAlmostEqual(e1,energy(centered(b)),places=11)
            result=transport(a,b)
            change=e1/energy(b)-energy(z)/energy(a)
            self.assertAlmostEqual(float(result['increment']),change,places=12)
            self.assertAlmostEqual(float(result['finite_cross']+result['quadratic']),change,places=12)

    def test_relative_concentration_without_absolute_collapse(self):
        z=np.array([[1.,-1.],[-1.,1.]])
        a=z+np.ones((2,2));b=z+1e4*np.ones((2,2))
        self.assertEqual(energy(centered(a)),energy(centered(b)))
        self.assertLess(energy(centered(b))/energy(b),1e-7)

    def test_absolute_collapse_without_relative_concentration(self):
        a=np.array([[1.,2.],[-1.,1.]])
        b=1e-8*a
        self.assertLess(energy(centered(b)),1e-14)
        self.assertAlmostEqual(energy(centered(a))/energy(a),energy(centered(b))/energy(b))

    def test_transverse_creation_is_quadratic_at_the_face(self):
        a=np.ones((2,2));b=a+.2*np.array([[1.,-1.],[-1.,1.]])
        r=transport(a,b)
        self.assertEqual(r['finite_cross'],0.)
        self.assertGreater(r['quadratic'],0.)
        self.assertAlmostEqual(r['quadratic'],r['increment'])

    def test_categorical_error_bound_is_shift_invariant(self):
        rng=np.random.default_rng(998)
        for v in [2,3,17]:
            for _ in range(32):
                z=rng.normal(size=v);d=rng.normal(size=v)
                lp=logsoftmax(z);lq=logsoftmax(z+d)
                kl=float(np.exp(lp)@(lp-lq))
                self.assertLessEqual(kl,energy(d)/4+1e-13)
                np.testing.assert_allclose(logsoftmax(z+50),lp,atol=1e-13)

    def test_temporal_interference_attains_both_signs(self):
        for ds in [np.array([[2.,-3.],[-2.,3.]]),np.ones((4,2))]:
            q=energy(ds);v=ds.sum(0)
            cross=2*sum(np.vdot(x,y) for i,x in enumerate(ds) for y in ds[i+1:])
            self.assertAlmostEqual(energy(v),q+cross)
            self.assertGreaterEqual(cross,-q)
            self.assertLessEqual(cross,(len(ds)-1)*q)
        self.assertEqual(energy(np.array([[2.,-3.],[-2.,3.]]).sum(0)),0.)

if __name__=='__main__':unittest.main()
