#!/usr/bin/env python
import unittest
from copy import deepcopy

import numpy as np

from simweights import (
    GenerationSurface,
    GenerationSurfaceCollection,
    NaturalRateCylinder,
    PDGCode,
    PowerLaw,
)


class TestGenerationSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1 = PowerLaw(-1, 10, 100)
        cls.p2 = PowerLaw(-2, 50, 500)
        cls.c1 = NaturalRateCylinder(3, 8, -1, 1)
        cls.c2 = NaturalRateCylinder(4, 8, -1, 1)

        cls.s0 = 10000 * GenerationSurface(2212, cls.p1, cls.c1)
        cls.s1 = 20000 * GenerationSurface(2212, cls.p1, cls.c1)
        cls.s2 = 10000 * GenerationSurface(2212, cls.p2, cls.c1)
        cls.s3 = 10000 * GenerationSurface(2212, cls.p1, cls.c2)
        cls.s4 = 10000 * GenerationSurface(2213, cls.p1, cls.c1)

        cls.gsc1 = GenerationSurfaceCollection(cls.s0, cls.s1)
        cls.gsc2 = GenerationSurfaceCollection(cls.s0, cls.s2)
        cls.gsc3 = GenerationSurfaceCollection(cls.s0, cls.s3)
        cls.gsc4 = GenerationSurfaceCollection(cls.s0, cls.s4)

    def test_compatible(self):
        assert self.s0.is_compatible(self.s0)
        assert self.s1.is_compatible(self.s1)
        assert self.s2.is_compatible(self.s2)
        assert self.s3.is_compatible(self.s3)
        assert self.s4.is_compatible(self.s4)

        assert self.s0.is_compatible(self.s1)
        assert not self.s0.is_compatible(self.s2)
        assert not self.s0.is_compatible(self.s3)
        assert not self.s0.is_compatible(self.s4)

        self.assertEqual(self.s0, self.s0)
        self.assertEqual(self.s1, self.s1)
        self.assertEqual(self.s2, self.s2)
        self.assertEqual(self.s3, self.s3)
        self.assertEqual(self.s4, self.s4)

        self.assertNotEqual(self.s0, self.s1)
        self.assertNotEqual(self.s0, self.s2)
        self.assertNotEqual(self.s0, self.s3)
        self.assertNotEqual(self.s0, self.s4)

    def test_get_pdgids(self):
        self.assertEqual(self.s0.get_pdgids(), [2212])
        self.assertEqual(self.s1.get_pdgids(), [2212])
        self.assertEqual(self.s2.get_pdgids(), [2212])
        self.assertEqual(self.s3.get_pdgids(), [2212])
        self.assertEqual(self.s4.get_pdgids(), [2213])
        self.assertEqual(self.gsc1.get_pdgids(), [2212])
        self.assertEqual(self.gsc2.get_pdgids(), [2212])
        self.assertEqual(self.gsc3.get_pdgids(), [2212])
        self.assertEqual(self.gsc4.get_pdgids(), [2212, 2213])

    def test_energy_range(self):
        self.assertEqual(self.s0.get_energy_range(2212), (10, 100))
        self.assertEqual(self.s2.get_energy_range(2212), (50, 500))
        self.assertEqual(self.s4.get_energy_range(2213), (10, 100))
        with self.assertRaises(AssertionError):
            self.s0.get_energy_range(2213)
        with self.assertRaises(AssertionError):
            self.s4.get_energy_range(2212)

        self.assertEqual(self.gsc1.get_energy_range(2212), (10, 100))
        self.assertEqual(self.gsc2.get_energy_range(2212), (10, 500))
        self.assertEqual(self.gsc4.get_energy_range(None), (10, 100))
        self.assertEqual(self.gsc4.get_energy_range(None), (10, 100))
        with self.assertRaises(AssertionError):
            self.gsc1.get_energy_range(2213)
        with self.assertRaises(AssertionError):
            self.gsc2.get_energy_range(2213)

    def test_cos_zenith_range(self):
        self.assertEqual(self.s0.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.s2.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.s4.get_cos_zenith_range(2213), (-1, 1))
        with self.assertRaises(AssertionError):
            self.s0.get_cos_zenith_range(2213)
        with self.assertRaises(AssertionError):
            self.s4.get_cos_zenith_range(2212)

        self.assertEqual(self.gsc1.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.gsc2.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.gsc4.get_cos_zenith_range(None), (-1, 1))
        self.assertEqual(self.gsc4.get_cos_zenith_range(None), (-1, 1))
        with self.assertRaises(AssertionError):
            self.gsc1.get_cos_zenith_range(2213)
        with self.assertRaises(AssertionError):
            self.gsc2.get_cos_zenith_range(2213)

    def test_addition(self):
        n0 = self.s0.nevents
        n1 = self.s1.nevents
        s = self.s0 + self.s1
        self.assertEqual(type(s), GenerationSurface)
        self.assertEqual(s.nevents, 30000)
        self.assertEqual(s.nevents, self.s0.nevents + self.s1.nevents)
        self.assertNotEqual(s, self.s0)
        self.assertNotEqual(s, self.s1)
        self.assertEqual(n0, self.s0.nevents)
        self.assertEqual(n1, self.s1.nevents)
        self.assertAlmostEqual(
            s.get_epdf(2212, 50, 0), self.s0.get_epdf(2212, 50, 0) + self.s1.get_epdf(2212, 50, 0)
        )

        ss = self.s0 + self.s2
        self.assertEqual(type(ss), GenerationSurfaceCollection)
        self.assertEqual(len(ss.spectra), 1)
        self.assertEqual(len(ss.spectra[2212]), 2)
        self.assertEqual(ss.spectra[2212][0], self.s0)
        self.assertEqual(ss.spectra[2212][1], self.s2)
        self.assertAlmostEqual(
            ss.get_epdf(2212, 50, 0), self.s0.get_epdf(2212, 50, 0) + self.s2.get_epdf(2212, 50, 0)
        )

        s3 = self.s0 + self.s3
        self.assertEqual(type(s3), GenerationSurfaceCollection)
        self.assertEqual(len(s3.spectra), 1)
        self.assertEqual(len(s3.spectra[2212]), 2)
        self.assertEqual(s3.spectra[2212][0], self.s0)
        self.assertEqual(s3.spectra[2212][1], self.s3)
        self.assertAlmostEqual(
            s3.get_epdf(2212, 50, 0), self.s0.get_epdf(2212, 50, 0) + self.s3.get_epdf(2212, 50, 0)
        )

        s4 = self.s0 + self.s4
        self.assertEqual(type(s4), GenerationSurfaceCollection)
        self.assertEqual(len(s4.spectra), 2)
        self.assertEqual(len(s4.spectra[2212]), 1)
        self.assertEqual(len(s4.spectra[2213]), 1)
        self.assertEqual(s4.spectra[2212][0], self.s0)
        self.assertEqual(s4.spectra[2213][0], self.s4)
        self.assertAlmostEqual(s4.get_epdf(2212, 50, 0), self.s0.get_epdf(2212, 50, 0))
        self.assertAlmostEqual(s4.get_epdf(2213, 50, 0), self.s4.get_epdf(2213, 50, 0))

        with self.assertRaises(TypeError):
            self.s0 + None

        with self.assertRaises(TypeError):
            self.s0 + int

        with self.assertRaises(TypeError):
            self.s0 + PowerLaw

    def test_multiplication(self):
        sa = deepcopy(self.s0)
        said = id(sa)
        sa *= 4.4
        self.assertEqual(said, id(sa))
        self.assertEqual(sa.nevents, 44000)
        self.assertEqual(self.s0.nevents, 10000)
        self.assertAlmostEqual(sa.get_epdf(2212, 50, 0), 4.4 * self.s0.get_epdf(2212, 50, 0))

        sb = self.s0 * 5.5
        self.assertNotEqual(id(sb), id(self.s0))
        self.assertEqual(sb.nevents, 55000)
        self.assertEqual(self.s0.nevents, 10000)
        self.assertAlmostEqual(sb.get_epdf(2212, 50, 0), 5.5 * self.s0.get_epdf(2212, 50, 0))

        sc = 6.6 * self.s0
        self.assertNotEqual(id(sc), id(self.s0))
        self.assertEqual(sc.nevents, 66000)
        self.assertEqual(self.s0.nevents, 10000)
        self.assertAlmostEqual(sc.get_epdf(2212, 50, 0), 6.6 * self.s0.get_epdf(2212, 50, 0))

    def test_repr(self):
        rep = repr(self.s0)
        nevents, s = rep.split("*")
        self.assertEqual(float(nevents), 1e4)
        s = s.strip()
        self.assertEqual(s[:18], "GenerationSurface(")
        self.assertEqual(s[-1:], ")")
        v = s[18:-1].split(",")
        self.assertEqual(v[0], "PPlus")
        self.assertEqual(",".join(v[1:4]).strip(), repr(self.p1))
        self.assertEqual(",".join(v[4:8]).strip(), repr(self.c1))

        PPlus = PDGCode.PPlus  # noqa: F841
        self.assertEqual(eval(repr(self.s0)), self.s0)
        self.assertEqual(eval(repr(self.s1)), self.s1)
        self.assertEqual(eval(repr(self.s2)), self.s2)
        self.assertEqual(eval(repr(self.s3)), self.s3)
        self.assertEqual(eval(repr(self.s4)), self.s4)

    def test_powerlaw(self):
        N = self.s0.nevents
        E = np.geomspace(self.p1.a, self.p1.b - 1 / N, N)
        cz = np.linspace(self.c1.cos_zen_min, self.c1.cos_zen_max, N)
        w = 1 / self.s0.get_epdf(2212, E, cz)

        area = (self.p1.b - self.p1.a) * (
            2 * self.c1.radius * np.pi ** 2 * (self.c1.radius + self.c1.length)
        )

        self.assertAlmostEqual(area, self.s0.get_surface_area())
        self.assertAlmostEqual(area, self.s0.energy_dist.span * self.s0.spatial_dist.etendue)
        self.assertAlmostEqual(w.sum() / area, 1, 4)

        self.assertEqual(self.s0.spatial_dist, self.c1)
        self.assertIsNot(self.s0.spatial_dist, self.c1)
        self.assertEqual(self.s0.energy_dist, self.p1)
        self.assertIsNot(self.s0.energy_dist, self.p1)

    def test_two_surfaces(self):
        N = self.s0.nevents
        cz = np.linspace(self.c1.cos_zen_min, self.c1.cos_zen_max, N)
        q = np.linspace(1 / 2 / N, 1 - 1 / 2 / N, N)
        E1 = 10 * np.exp(q * np.log(100 / 10))
        E2 = (q * (500 ** -1 - 50 ** -1) + 50 ** -1) ** -1

        surf = self.s0 + self.s2
        E = np.r_[E1, E2]
        czc = np.r_[cz, cz]
        wc = 1 / surf.get_epdf(2212, E, czc)

        self.assertAlmostEqual(wc.sum() / (self.p2.b - self.p1.a) / self.c1.etendue, 1, 3)

        self.assertEqual(self.s0.spatial_dist, self.c1)
        self.assertIsNot(self.s0.spatial_dist, self.c1)
        self.assertEqual(self.s0.energy_dist, self.p1)
        self.assertIsNot(self.s0.energy_dist, self.p1)

        self.assertEqual(self.s2.spatial_dist, self.c1)
        self.assertIsNot(self.s2.spatial_dist, self.c1)
        self.assertEqual(self.s2.energy_dist, self.p2)
        self.assertIsNot(self.s2.energy_dist, self.p2)

        self.assertEqual(len(surf.spectra), 1)
        np.testing.assert_array_equal(list(surf.spectra.keys()), [2212])

        self.assertEqual(surf.spectra[2212][0], self.s0)
        self.assertIsNot(surf.spectra[2212][0], self.s0)
        self.assertEqual(surf.spectra[2212][0].spatial_dist, self.s0.spatial_dist)
        self.assertIsNot(surf.spectra[2212][0].spatial_dist, self.s0.spatial_dist)
        self.assertEqual(surf.spectra[2212][0].energy_dist, self.s0.energy_dist)
        self.assertIsNot(surf.spectra[2212][0].energy_dist, self.s0.energy_dist)

        self.assertEqual(surf.spectra[2212][1], self.s2)
        self.assertIsNot(surf.spectra[2212][1], self.s2)
        self.assertEqual(surf.spectra[2212][1].spatial_dist, self.s2.spatial_dist)
        self.assertIsNot(surf.spectra[2212][1].spatial_dist, self.s2.spatial_dist)
        self.assertEqual(surf.spectra[2212][1].energy_dist, self.s2.energy_dist)
        self.assertIsNot(surf.spectra[2212][1].energy_dist, self.s2.energy_dist)

    def test_instantiation(self):
        s02 = GenerationSurfaceCollection(self.s0, self.s0)
        self.assertEqual(len(s02.spectra), 1)
        self.assertEqual(len(s02.spectra[2212]), 1)
        assert s02.spectra[2212][0].is_compatible(self.s0)
        self.assertEqual(s02.spectra[2212][0].nevents, 20000)

        s02 = GenerationSurfaceCollection(self.s0, self.s2)
        self.assertEqual(len(s02.spectra), 1)
        self.assertEqual(len(s02.spectra[2212]), 2)
        assert s02.spectra[2212][0].is_compatible(self.s0)
        assert s02.spectra[2212][1].is_compatible(self.s2)
        self.assertEqual(s02.spectra[2212][0].nevents, 10000)
        self.assertEqual(s02.spectra[2212][1].nevents, 10000)

        s04 = GenerationSurfaceCollection(self.s0, self.s4)
        self.assertEqual(len(s04.spectra), 2)
        self.assertEqual(len(s04.spectra[2212]), 1)
        self.assertEqual(len(s04.spectra[2213]), 1)
        assert s04.spectra[2212][0].is_compatible(self.s0)
        assert s04.spectra[2213][0].is_compatible(self.s4)
        self.assertEqual(s04.spectra[2212][0].nevents, 10000)
        self.assertEqual(s04.spectra[2213][0].nevents, 10000)

        with self.assertRaises(AssertionError):
            GenerationSurfaceCollection(self.s1, None)
        with self.assertRaises(AssertionError):
            GenerationSurfaceCollection(self.s1, self.p1)
        with self.assertRaises(AssertionError):
            GenerationSurfaceCollection(5)

    def test_addition_gsc(self):
        s0 = self.gsc1 + self.s0
        self.assertEqual(type(s0), GenerationSurfaceCollection)
        self.assertEqual(len(s0.spectra), 1)
        self.assertEqual(len(s0.spectra[2212]), 1)
        assert s0.spectra[2212][0].is_compatible(self.s0)
        self.assertEqual(s0.spectra[2212][0].nevents, 40000)

        s2 = self.gsc2 + self.s0
        self.assertEqual(type(s2), GenerationSurfaceCollection)
        self.assertEqual(len(s2.spectra), 1)
        self.assertEqual(len(s2.spectra[2212]), 2)
        assert s2.spectra[2212][0].is_compatible(self.s0)
        assert s2.spectra[2212][1].is_compatible(self.s2)
        self.assertEqual(s2.spectra[2212][0].nevents, 20000)
        self.assertEqual(s2.spectra[2212][1].nevents, 10000)

        s4 = self.gsc4 + self.s0
        self.assertEqual(type(s4), GenerationSurfaceCollection)
        self.assertEqual(len(s4.spectra), 2)
        self.assertEqual(len(s4.spectra[2212]), 1)
        self.assertEqual(len(s4.spectra[2213]), 1)
        assert s4.spectra[2212][0].is_compatible(self.s0)
        assert s4.spectra[2213][0].is_compatible(self.s4)
        self.assertEqual(s4.spectra[2212][0].nevents, 20000)
        self.assertEqual(s4.spectra[2213][0].nevents, 10000)

        s5 = self.gsc1 + self.gsc2
        self.assertEqual(type(s5), GenerationSurfaceCollection)
        self.assertEqual(len(s5.spectra), 1)
        self.assertEqual(len(s5.spectra[2212]), 2)
        assert s5.spectra[2212][0].is_compatible(self.s0)
        assert s5.spectra[2212][1].is_compatible(self.s2)
        self.assertEqual(s5.spectra[2212][0].nevents, 40000)
        self.assertEqual(s5.spectra[2212][1].nevents, 10000)

        s6 = self.gsc2 + self.gsc4
        self.assertEqual(type(s6), GenerationSurfaceCollection)
        self.assertEqual(len(s6.spectra), 2)
        self.assertEqual(len(s6.spectra[2212]), 2)
        self.assertEqual(len(s6.spectra[2213]), 1)
        assert s6.spectra[2212][0].is_compatible(self.s0)
        assert s6.spectra[2212][1].is_compatible(self.s2)
        assert s6.spectra[2213][0].is_compatible(self.s4)
        self.assertEqual(s6.spectra[2212][0].nevents, 20000)
        self.assertEqual(s6.spectra[2212][1].nevents, 10000)
        self.assertEqual(s6.spectra[2213][0].nevents, 10000)

        with self.assertRaises(ValueError):
            self.gsc2 + None
        with self.assertRaises(ValueError):
            self.gsc2 + self.p1
        with self.assertRaises(ValueError):
            self.gsc2 + self.c1
        with self.assertRaises(ValueError):
            self.gsc2 + 5

    def test_multiplication_gsc(self):
        s0 = deepcopy(self.gsc2)
        s0 *= 2.3
        self.assertEqual(s0.spectra[2212][0], 2.3 * self.s0)
        self.assertEqual(s0.spectra[2212][1], 2.3 * self.s2)

        s1 = 3.4 * self.gsc2
        self.assertEqual(s1.spectra[2212][0], 3.4 * self.s0)
        self.assertEqual(s1.spectra[2212][1], 3.4 * self.s2)

        s1 = self.gsc3 * 4.5
        self.assertEqual(s1.spectra[2212][0], 4.5 * self.s0)
        self.assertEqual(s1.spectra[2212][1], 4.5 * self.s3)

    def test_equal_gsc(self):
        self.assertEqual(self.gsc1, self.gsc1)
        self.assertEqual(self.gsc2, self.gsc2)
        self.assertEqual(self.gsc3, self.gsc3)
        self.assertEqual(self.gsc4, self.gsc4)
        self.assertNotEqual(self.gsc1, self.gsc2)
        self.assertNotEqual(self.gsc1, self.gsc3)
        self.assertNotEqual(self.gsc1, self.gsc4)
        self.assertNotEqual(self.gsc2, self.gsc3)
        self.assertNotEqual(self.gsc2, self.gsc4)
        self.assertNotEqual(self.gsc3, self.gsc4)
        self.assertEqual(self.gsc1, GenerationSurfaceCollection(self.s1, self.s0))
        self.assertEqual(self.gsc2, GenerationSurfaceCollection(self.s2, self.s0))
        self.assertEqual(self.gsc3, GenerationSurfaceCollection(self.s3, self.s0))
        self.assertEqual(self.gsc4, GenerationSurfaceCollection(self.s4, self.s0))

    def test_repr_gsc(self):
        PPlus = PDGCode.PPlus  # noqa: F841
        self.assertEqual(self.gsc1, eval(repr(self.gsc1)))
        self.assertEqual(self.gsc2, eval(repr(self.gsc2)))
        self.assertEqual(self.gsc3, eval(repr(self.gsc3)))
        self.assertEqual(self.gsc4, eval(repr(self.gsc4)))

        s = str(self.gsc2 + self.gsc3 + self.gsc4).split("\n")
        self.assertEqual(s[0], "< GenerationSurfaceCollection")
        self.assertEqual(eval(s[1].split()[-1]), self.c1)
        self.assertEqual(eval(s[1].split()[-2]), self.p1)
        self.assertEqual(eval(s[2].split()[-1]), self.c1)
        self.assertEqual(eval(s[2].split()[-2]), self.p2)
        self.assertEqual(eval(s[3].split()[-1]), self.c2)
        self.assertEqual(eval(s[3].split()[-2]), self.p1)
        self.assertEqual(eval(s[4].split()[-1]), self.c1)
        self.assertEqual(eval(s[4].split()[-2]), self.p1)
        self.assertEqual(s[5], ">")


if __name__ == "__main__":
    unittest.main()
