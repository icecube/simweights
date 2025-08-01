#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest
from copy import deepcopy

from simweights import NaturalRateCylinder, PDGCode, PowerLaw
from simweights._generation_surface import CompositeSurface, GenerationSurface


class Testgeneration_surface(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.p1 = PowerLaw(-1, 10, 100)
        cls.p2 = PowerLaw(-2, 50, 500)
        cls.c1 = NaturalRateCylinder(3, 8, -1, 1)
        cls.c2 = NaturalRateCylinder(4, 8, -1, 1)

        cls.N1 = 10000
        cls.N2 = 20000
        cls.s0 = GenerationSurface(2212, cls.N1, cls.p1, cls.c1)
        cls.s1 = GenerationSurface(2212, cls.N2, cls.p1, cls.c1)
        cls.s2 = GenerationSurface(2212, cls.N1, cls.p2, cls.c1)
        cls.s3 = GenerationSurface(2212, cls.N1, cls.p1, cls.c2)
        cls.s4 = GenerationSurface(22, cls.N1, cls.p1, cls.c1)

        cls.gsc1 = CompositeSurface(cls.s0, cls.s1)
        cls.gsc2 = CompositeSurface(cls.s0, cls.s2)
        cls.gsc3 = CompositeSurface(cls.s0, cls.s3)
        cls.gsc4 = CompositeSurface(cls.s0, cls.s4)

    def test_compatible(self):
        self.assertEqual(self.s0, self.s0)
        self.assertEqual(self.s1, self.s1)
        self.assertEqual(self.s2, self.s2)
        self.assertEqual(self.s3, self.s3)
        self.assertEqual(self.s4, self.s4)

        self.assertNotEqual(self.s0, self.s1)
        self.assertNotEqual(self.s0, self.s2)
        self.assertNotEqual(self.s0, self.s3)
        self.assertNotEqual(self.s0, self.s4)

        self.assertNotEqual(self.s0, None)
        self.assertNotEqual(self.s0, 47)
        self.assertNotEqual(self.s0, object())
        self.assertNotEqual(self.s0, self.p1)

        with self.assertRaises(NotImplementedError):
            self.s0.get_epdf({})
        with self.assertRaises(NotImplementedError):
            self.s1.get_epdf({})
        with self.assertRaises(NotImplementedError):
            self.s3.get_epdf({})
        with self.assertRaises(NotImplementedError):
            self.s4.get_epdf({})

    def test_energy_range(self):
        self.assertEqual(self.s0.get_energy_range(2212), (10, 100))
        self.assertEqual(self.s2.get_energy_range(2212), (50, 500))
        self.assertEqual(self.s4.get_energy_range(22), (10, 100))
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
            self.gsc2.get_energy_range(22)

    def test_cos_zenith_range(self):
        self.assertEqual(self.s0.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.s2.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.s4.get_cos_zenith_range(22), (-1, 1))
        with self.assertRaises(AssertionError):
            self.s0.get_cos_zenith_range(2213)
        with self.assertRaises(AssertionError):
            self.s4.get_cos_zenith_range(2212)

        self.assertEqual(self.gsc1.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.gsc2.get_cos_zenith_range(2212), (-1, 1))
        self.assertEqual(self.gsc4.get_cos_zenith_range(None), (-1, 1))
        self.assertEqual(self.gsc4.get_cos_zenith_range(None), (-1, 1))
        with self.assertRaises(AssertionError):
            self.gsc1.get_cos_zenith_range(22)
        with self.assertRaises(AssertionError):
            self.gsc2.get_cos_zenith_range(22)

    def test_addition(self):
        n0 = 10000
        n1 = 20000
        s = CompositeSurface(self.s0, self.s1)
        self.assertEqual(type(s), CompositeSurface)
        self.assertEqual(s.components[2212][0].nevents, 30000)
        self.assertNotEqual(s, self.s0)
        self.assertNotEqual(s, self.s1)
        self.assertEqual(n0, self.s0.nevents)
        self.assertEqual(n1, self.s1.nevents)

        ss = CompositeSurface(self.s0, self.s2)
        self.assertEqual(type(ss), CompositeSurface)
        self.assertEqual(len(ss.components), 1)
        self.assertEqual(len(ss.components[2212]), 2)
        self.assertEqual(ss.components[2212][0], self.s0)
        self.assertEqual(ss.components[2212][1], self.s2)

        s3 = CompositeSurface(self.s0, self.s3)
        self.assertEqual(type(s3), CompositeSurface)
        self.assertEqual(len(s3.components), 1)
        self.assertEqual(len(s3.components[2212]), 2)
        self.assertEqual(s3.components[2212][0], self.s0)
        self.assertEqual(s3.components[2212][1], self.s3)

        s4 = CompositeSurface(self.s0, self.s4)
        self.assertEqual(type(s4), CompositeSurface)
        self.assertEqual(len(s4.components), 2)
        self.assertEqual(len(s4.components[2212]), 1)
        self.assertEqual(len(s4.components[22]), 1)
        self.assertEqual(s4.components[2212][0], self.s0)
        self.assertEqual(s4.components[22][0], self.s4)

        with self.assertRaises(TypeError):
            CompositeSurface(self.s0, 47)
        with self.assertRaises(TypeError):
            self.s0 + None
        with self.assertRaises(TypeError):
            self.s0 + int
        with self.assertRaises(TypeError):
            self.s0 + self.p1

        with self.assertRaises(TypeError):
            47 + self.s0
        with self.assertRaises(TypeError):
            None + self.s0
        with self.assertRaises(TypeError):
            int + self.s0
        with self.assertRaises(TypeError):
            self.p1 + self.s0

    def test_multiplication(self):
        sa = deepcopy(self.s0)
        sa.scale(4.4)
        self.assertEqual(sa.nevents, 44000)
        self.assertEqual(self.s0.nevents, 10000)

        sb = deepcopy(self.s0)
        sb.scale(5.5)
        self.assertNotEqual(id(sb), id(self.s0))
        self.assertEqual(sb.nevents, 55000)
        self.assertEqual(self.s0.nevents, 10000)

    def test_repr(self):
        Gamma = PDGCode.Gamma  # noqa: F841
        PPlus = PDGCode.PPlus  # noqa: F841
        self.assertEqual(eval(repr(self.s0)), self.s0)
        self.assertEqual(eval(repr(self.s1)), self.s1)
        self.assertEqual(eval(repr(self.s2)), self.s2)
        self.assertEqual(eval(repr(self.s3)), self.s3)
        self.assertEqual(eval(repr(self.s4)), self.s4)

    def test_addition_gsc(self):
        s0 = CompositeSurface(self.gsc1, self.s0)
        self.assertEqual(type(s0), CompositeSurface)
        self.assertEqual(len(s0.components), 1)
        self.assertEqual(len(s0.components[2212]), 1)
        self.assertEqual(s0.components[2212][0].nevents, 40000)

        s2 = CompositeSurface(self.gsc2, self.s0)
        self.assertEqual(type(s2), CompositeSurface)
        self.assertEqual(len(s2.components), 1)
        self.assertEqual(len(s2.components[2212]), 2)
        self.assertEqual(s2.components[2212][0].nevents, 20000)
        self.assertEqual(s2.components[2212][1].nevents, 10000)

        s4 = CompositeSurface(self.gsc4, self.s0)
        self.assertEqual(type(s4), CompositeSurface)
        self.assertEqual(len(s4.components), 2)
        self.assertEqual(len(s4.components[2212]), 1)
        self.assertEqual(len(s4.components[22]), 1)
        self.assertEqual(s4.components[2212][0].nevents, 20000)
        self.assertEqual(s4.components[22][0].nevents, 10000)

        s5 = CompositeSurface(self.gsc1, self.gsc2)
        self.assertEqual(type(s5), CompositeSurface)
        self.assertEqual(len(s5.components), 1)
        self.assertEqual(len(s5.components[2212]), 2)
        self.assertEqual(s5.components[2212][0].nevents, 40000)
        self.assertEqual(s5.components[2212][1].nevents, 10000)

        s6 = CompositeSurface(self.gsc2, self.gsc4)
        self.assertEqual(type(s6), CompositeSurface)
        self.assertEqual(len(s6.components), 2)
        self.assertEqual(len(s6.components[2212]), 2)
        self.assertEqual(len(s6.components[22]), 1)
        self.assertEqual(s6.components[2212][0].nevents, 20000)
        self.assertEqual(s6.components[2212][1].nevents, 10000)
        self.assertEqual(s6.components[22][0].nevents, 10000)

        a = CompositeSurface(self.gsc1, self.gsc2, self.gsc3, self.gsc4)
        b = CompositeSurface()
        b.insert(self.gsc1)
        b.insert(self.gsc2)
        b.insert(self.gsc3)
        b.insert(self.gsc4)
        self.assertEqual(a, b)

        with self.assertRaises(TypeError):
            self.gsc2.insert(47)
        with self.assertRaises(TypeError):
            self.gsc2.insert(None)
        with self.assertRaises(TypeError):
            self.gsc2.insert(int)
        with self.assertRaises(TypeError):
            self.gsc2.insert(self.p1)
        with self.assertRaises(TypeError):
            self.gsc2.insert(self.c1)

    def test_equal_gsc(self):
        self.assertNotEqual(None, self.gsc1)
        self.assertNotEqual([], self.gsc1)
        self.assertNotEqual(self.c1, self.gsc1)
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
        self.assertEqual(self.gsc1, CompositeSurface(self.s1, self.s0))
        self.assertEqual(self.gsc2, CompositeSurface(self.s2, self.s0))
        self.assertEqual(self.gsc3, CompositeSurface(self.s3, self.s0))
        self.assertEqual(self.gsc4, CompositeSurface(self.s4, self.s0))

    def test_repr_gsc(self):
        PPlus = PDGCode.PPlus  # noqa: F841
        Gamma = PDGCode.Gamma  # noqa: F841
        self.assertEqual(self.gsc1, eval(repr(self.gsc1)))
        self.assertEqual(self.gsc2, eval(repr(self.gsc2)))
        self.assertEqual(self.gsc3, eval(repr(self.gsc3)))
        self.assertEqual(self.gsc4, eval(repr(self.gsc4)))

        s = str(CompositeSurface(self.gsc2, self.gsc3, self.gsc4)).split("\n")
        self.assertEqual(s[0], "< CompositeSurface")
        self.assertEqual(eval("".join(s[1].split()[-4:])[:-1]), self.c1)
        self.assertEqual(eval("".join(s[1].split()[-7:-4])[:-1]), self.p1)
        self.assertEqual(eval("".join(s[2].split()[-4:])[:-1]), self.c1)
        self.assertEqual(eval("".join(s[2].split()[-7:-4])[:-1]), self.p2)
        self.assertEqual(eval("".join(s[3].split()[-4:])[:-1]), self.c2)
        self.assertEqual(eval("".join(s[3].split()[-7:-4])[:-1]), self.p1)
        self.assertEqual(eval("".join(s[4].split()[-4:])[:-1]), self.c1)
        self.assertEqual(eval("".join(s[4].split()[-7:-4])[:-1]), self.p1)
        self.assertEqual(s[5], ">")


if __name__ == "__main__":
    unittest.main()
