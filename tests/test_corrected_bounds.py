"""Independent finite examples that distinguish exact criteria from envelopes."""
import unittest
import numpy as np

class CorrectedBoundTests(unittest.TestCase):
    def test_transport_covariance_matches_enumerated_law(self):
        # All sign sequences equally weighted: nonnormal, noncommuting transport.
        from itertools import product
        matrices=[np.array([[1.,2.],[0.,.5]]),np.array([[.2,0.],[1.,1.]])]
        bias=[np.array([-.5,1.]),np.array([.1,-.2])]
        initial=np.array([1.,-1.]);directions=[np.array([0.,.3]),np.array([.2,.1])]
        states=[]
        for signs in product([-1,1],repeat=2):
            z=initial.copy()
            for a,b,d,e in zip(matrices,bias,directions,signs):z=a@z+b+e*d
            states.append(z)
        mu=matrices[1]@(matrices[0]@initial+bias[0])+bias[1]
        variance=np.sum((matrices[1]@directions[0])**2)+np.sum(directions[1]**2)
        np.testing.assert_allclose(np.mean(states,axis=0),mu,atol=1e-14)
        self.assertAlmostEqual(np.mean(np.sum(np.array(states)**2,axis=1)),mu@mu+variance)
    def test_mean_components_cancel(self):
        self.assertEqual(1+(-1),0)
        self.assertGreater(abs(1)+abs(-1),0)
    def test_killed_noise_has_nonvanishing_norm_envelope(self):
        a=np.diag([1.,0.]);noise=np.array([0.,1.])
        self.assertEqual(float(np.linalg.norm(a@noise)**2),0.)
        self.assertEqual(float(np.linalg.norm(a,2)**2*(noise@noise)),1.)
    def test_radial_rescaling_has_zero_error_positive_budget(self):
        a=np.array([[1.,2.],[-1.,1.]])
        def u(x):return np.sum((x-x.mean(0))**2)/np.sum(x*x)
        self.assertAlmostEqual(u(1.5*a),u(a))
        self.assertEqual((3*.5**2+.5**3)/(1-.5)**2,3.5)

if __name__=='__main__':unittest.main()
