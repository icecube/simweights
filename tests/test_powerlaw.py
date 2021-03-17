#!/usr/bin/env python 
import unittest
import numpy as np
from scipy import stats
from scipy.integrate import quad
from simweights.powerlaw import PowerLaw

class TestPowerLaw(unittest.TestCase):
    def cmp_scipy(self,g,s):
        x = np.linspace(-1, s + 1, 10 * (s + 2) + 1)
        q = np.linspace(-1, 2, 51)

        p0 = stats.powerlaw(g + 1, scale=s)
        v0 = p0.pdf(x)
        c0 = p0.cdf(x)
        x0 = p0.ppf(q)

        p1 = PowerLaw(g, 0, s)
        v1 = p1.pdf(x)
        c1 = p1.cdf(x)
        x1 = p1.ppf(q)

        assert(0 in x)
        assert(s in x)
        assert np.allclose(v0, v1)
        assert np.allclose(c0, c1)
        assert np.allclose(x0, x1, equal_nan=True)

    def test_scipy(self):    
        self.cmp_scipy(0.0, 1)
        self.cmp_scipy(0.5, 2)
        self.cmp_scipy(1.0, 3)
        self.cmp_scipy(1.5, 1)
        self.cmp_scipy(2.0, 2)
        self.cmp_scipy(2.5, 3)
        
    def check_vals(self,p):
        x= np.linspace(p.a, p.b, 51)
        v1 = p.pdf(x) * x**-p.g
        assert np.allclose(v1, v1[0])
        
        c0 = [ quad(p.pdf, p.a, xx)[0] for xx in x ]
        c1 = p.cdf(x)
        assert np.allclose(c0, c1)
        assert np.allclose(p.ppf(c1), x)
        self.assertEqual(c1[0], 0)
        self.assertEqual(c1[-1], 1)

        x1 = np.linspace(p.b, p.b+2, 21)
        assert np.all(p.pdf(x1[1:]) == 0)
        assert np.all(p.cdf(x1) == 1)
        
        x2 = np.linspace(p.a - 2, p.a, 21)
        assert np.all(p.pdf(x2[:-1]) == 0)
        assert np.all(p.cdf(x2) == 0)
        
        assert np.all(np.isnan(p.ppf(np.linspace(-1, 0, 21)[:-1])))
        assert np.all(np.isnan(p.ppf(np.linspace( 1, 2, 21)[1:])))
        assert p.pdf(np.nan)==0
        assert np.isnan(p.cdf(np.nan))
        assert np.isnan(p.ppf(np.nan)) 


    def test_vals(self):
        self.check_vals(PowerLaw( 2, 1, 2))
        self.check_vals(PowerLaw( 1, 2, 3))
        self.check_vals(PowerLaw( 0, 3, 4))
        self.check_vals(PowerLaw(-1, 1, 2))
        self.check_vals(PowerLaw(-2, 2, 3))
        self.check_vals(PowerLaw(-3, 4, 5))
        self.check_vals(PowerLaw(-4, 5, 6))

    def test_rvs(self):
        p =  PowerLaw(-1, 1, 10)
        self.assertEqual(type(p.rvs()), np.float64)
        x1 = p.rvs(1)
        self.assertEqual(type(x1), np.ndarray)
        self.assertEqual(x1.shape, (1,))
        x2 = p.rvs(10)
        self.assertEqual(type(x2), np.ndarray)
        self.assertEqual(x2.shape, (10,))
        x3 = p.rvs((6, 8))
        self.assertEqual(type(x3), np.ndarray)
        self.assertEqual(x3.shape, (6, 8))

    def test_equal_operator(self):
        p1 = PowerLaw(-1, 1, 10)
        self.assertEqual(p1,PowerLaw(-1, 1, 10))
        self.assertNotEqual(p1,PowerLaw(-2, 1, 10))
        self.assertNotEqual(p1,PowerLaw(-1, 2, 10))
        self.assertNotEqual(p1,PowerLaw(-1, 1, 11))

if __name__ == '__main__':
    unittest.main()

