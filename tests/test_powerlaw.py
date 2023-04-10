#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
from scipy import stats
from scipy.integrate import quad
from simweights import PowerLaw


class TestPowerLaw(unittest.TestCase):
    def cmp_scipy(self, g, s):
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

        self.assertIn(0, x)
        self.assertIn(s, x)
        np.testing.assert_allclose(v0, v1)
        np.testing.assert_allclose(c0, c1)
        np.testing.assert_allclose(x0, x1)

        rep = str(p1)
        self.assertEqual(rep[:9], "PowerLaw(")
        self.assertEqual(rep[-1:], ")")
        v = rep[9:-1].split(",")
        self.assertEqual(float(v[0]), g)
        self.assertEqual(float(v[1]), 0)
        self.assertEqual(float(v[2]), s)

    def test_scipy(self):
        self.cmp_scipy(0.0, 1)
        self.cmp_scipy(0.5, 2)
        self.cmp_scipy(1.0, 3)
        self.cmp_scipy(1.5, 1)
        self.cmp_scipy(2.0, 2)
        self.cmp_scipy(2.5, 3)

    def check_vals(self, p):
        x = np.linspace(p.a, p.b, 51)
        v1 = p.pdf(x) * x**-p.g
        np.testing.assert_allclose(v1, v1[0])

        c0 = [quad(p.pdf, p.a, xx)[0] for xx in x]
        c1 = p.cdf(x)
        np.testing.assert_allclose(c0, c1)
        np.testing.assert_allclose(p.ppf(c1), x)
        self.assertEqual(c1[0], 0)
        self.assertEqual(c1[-1], 1)

        x1 = np.linspace(p.b, p.b + 2, 21)
        np.testing.assert_array_equal(p.pdf(x1[1:]), 0)
        np.testing.assert_array_equal(p.cdf(x1), 1)

        x2 = np.linspace(p.a - 2, p.a, 21)
        np.testing.assert_array_equal(p.pdf(x2[:-1]), 0)
        np.testing.assert_array_equal(p.cdf(x2), 0)

        np.testing.assert_array_equal(p.ppf(np.linspace(-1, 0, 21)[:-1]), np.nan)
        np.testing.assert_array_equal(p.ppf(np.linspace(1, 2, 21)[1:]), np.nan)
        np.testing.assert_array_equal(p.pdf(np.nan), 0)
        np.testing.assert_array_equal(p.cdf(np.nan), np.nan)
        np.testing.assert_array_equal(p.ppf(np.nan), np.nan)

        rep = str(p)
        self.assertEqual(rep[:9], "PowerLaw(")
        self.assertEqual(rep[-1:], ")")
        v = rep[9:-1].split(",")
        self.assertEqual(float(v[0]), p.g)
        self.assertEqual(float(v[1]), p.a)
        self.assertEqual(float(v[2]), p.b)

    def test_vals(self):
        self.check_vals(PowerLaw(2, 1, 2))
        self.check_vals(PowerLaw(1, 2, 3))
        self.check_vals(PowerLaw(0, 3, 4))
        self.check_vals(PowerLaw(-1, 1, 2))
        self.check_vals(PowerLaw(-2, 2, 3))
        self.check_vals(PowerLaw(-3, 4, 5))
        self.check_vals(PowerLaw(-4, 5, 6))

    def test_rvs(self):
        p = PowerLaw(-1, 1, 10)
        x0 = p.rvs()
        self.assertEqual(type(x0), np.ndarray)
        self.assertEqual(x0.shape, ())
        x1 = p.rvs(1)
        self.assertEqual(type(x1), np.ndarray)
        self.assertEqual(x1.shape, (1,))
        x2 = p.rvs(10)
        self.assertEqual(type(x2), np.ndarray)
        self.assertEqual(x2.shape, (10,))
        x3 = p.rvs((6, 8))
        self.assertEqual(type(x3), np.ndarray)
        self.assertEqual(x3.shape, (6, 8))

        x4 = p.rvs(None, np.random.RandomState())
        self.assertEqual(type(x4), np.ndarray)
        self.assertEqual(x4.shape, ())
        x5 = p.rvs(1, np.random.Generator(np.random.PCG64()))
        self.assertEqual(type(x5), np.ndarray)
        self.assertEqual(x5.shape, (1,))

        with self.assertRaises(ValueError):
            p.rvs(1, object())

    def test_equal_operator(self):
        p1 = PowerLaw(-1, 1, 10)
        self.assertEqual(p1, PowerLaw(-1, 1, 10))
        self.assertNotEqual(p1, PowerLaw(-2, 1, 10))
        self.assertNotEqual(p1, PowerLaw(-1, 2, 10))
        self.assertNotEqual(p1, PowerLaw(-1, 1, 11))

    def check_sample(self, g):
        p = PowerLaw(g, 1, 1000)
        N = 100000
        x = p.ppf(np.linspace(1 / 2 / N, 1 - 1 / 2 / N, N))
        w = 1 / p.pdf(x)
        self.assertAlmostEqual(w.sum() / N / p.span, 1, 2)

    def test_sample(self):
        self.check_sample(-0.5)
        self.check_sample(-1)
        self.check_sample(-1.5)
        self.check_sample(-2)
        self.check_sample(-2.5)

    def test_raises(self):
        p = PowerLaw(1, 1, 1000)
        with self.assertRaises(TypeError):
            p == object()  # noqa: B015
        with self.assertRaises(TypeError):
            p == np.array([])  # noqa: B015


if __name__ == "__main__":
    unittest.main()
