#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
from simweights import CircleInjector, NaturalRateCylinder, UniformSolidAngleCylinder
from simweights._spatial import CylinderBase


class TestSpatial(unittest.TestCase):
    def check_diff_etendue(self, c, le, r):
        le *= 1e2
        r *= 1e2

        self.assertAlmostEqual(c.projected_area(-1), np.pi * r**2, 5)
        self.assertAlmostEqual(c.projected_area(-0.5), np.pi * r**2 / 2 + 3**0.5 * le * r, 5)
        self.assertAlmostEqual(c.projected_area(-1 / 2**0.5), r / 2**0.5 * (np.pi * r + 2 * le), 5)
        self.assertAlmostEqual(c.projected_area(0), 2 * le * r)
        self.assertAlmostEqual(c.projected_area(1 / 2**0.5), r / 2**0.5 * (np.pi * r + 2 * le), 5)
        self.assertAlmostEqual(c.projected_area(0.5), np.pi * r**2 / 2 + 3**0.5 * le * r, 5)
        self.assertAlmostEqual(c.projected_area(1), np.pi * r**2, 5)

        with self.assertRaises(AssertionError):
            c.projected_area(-1.01)
        with self.assertRaises(AssertionError):
            c.projected_area(1.01)

        self.assertAlmostEqual(c._diff_etendue(-1), -np.pi**2 * r * (r + 2 * le), 4)
        self.assertAlmostEqual(
            c._diff_etendue(-0.5),
            -np.pi**2 / 4 * r * (r + 2 / 3 * le * (3**1.5 / np.pi + 8)),
            4,
        )
        self.assertAlmostEqual(
            c._diff_etendue(-(0.5**0.5)),
            -np.pi**2 / 2 * r * (r + le * (2 / np.pi + 3)),
            4,
        )
        self.assertAlmostEqual(c._diff_etendue(0), -np.pi**2 * le * r)
        self.assertAlmostEqual(
            c._diff_etendue(0.5**0.5),
            np.pi**2 / 2 * r * (r + le * (2 / np.pi - 1)),
            5,
        )
        self.assertAlmostEqual(
            c._diff_etendue(0.5),
            np.pi**2 / 4 * r * (r + 2 / 3 * le * (3**1.5 / np.pi - 4)),
            4,
        )
        self.assertAlmostEqual(c._diff_etendue(1), (np.pi * r) ** 2)

        with self.assertRaises(AssertionError):
            c._diff_etendue(np.nextafter(-1, -np.inf))
        with self.assertRaises(AssertionError):
            c._diff_etendue(np.nextafter(1, np.inf))

    def check_pdf_etendue(self, c, etendue):
        etendue *= 1e4
        self.assertAlmostEqual(c.etendue, etendue, 4)

        x = np.linspace(c.cos_zen_min, c.cos_zen_max)
        np.testing.assert_allclose(c.pdf(x), 1 / etendue, 1e-15)

        x = np.linspace(-2, np.nextafter(c.cos_zen_min, -np.inf))
        np.testing.assert_array_equal(c.pdf(x), 0)

        x = np.linspace(np.nextafter(c.cos_zen_max, np.inf), 2)
        np.testing.assert_array_equal(c.pdf(x), 0)

    def test_cylinder_base(self):
        length = 600
        radius = 400
        with self.assertRaises(ValueError):
            CylinderBase(length, radius, -1.5, 0.5)
        with self.assertRaises(ValueError):
            CylinderBase(length, radius, -0.5, 1.5)
        with self.assertRaises(ValueError):
            CylinderBase(length, radius, 0.5, -0.5)
        c = CylinderBase(length, radius, -1, 1)
        self.check_diff_etendue(c, length, radius)
        with self.assertRaises(NotImplementedError):
            c.pdf(0.5)
        self.assertEqual(c, c)

    def test_natural_rate_cylinder(self):
        last_c1 = None
        for le in range(100, 1000, 300):
            for r in range(100, 1000, 300):
                c1 = NaturalRateCylinder(le, r, -1, 1)
                self.check_diff_etendue(c1, le, r)
                self.check_pdf_etendue(c1, 2 * np.pi**2 * r * (r + le))

                c2 = NaturalRateCylinder(le, r, -1, 0)
                self.check_pdf_etendue(c2, np.pi**2 * r * (r + le))

                c3 = NaturalRateCylinder(le, r, 0, 1)
                self.check_pdf_etendue(c3, np.pi**2 * r * (r + le))

                self.assertEqual(c1, c1)
                self.assertNotEqual(c1, c2)
                self.assertNotEqual(c1, c3)
                self.assertEqual(c2, c2)
                self.assertNotEqual(c2, c3)
                self.assertEqual(c3, c3)
                self.assertNotEqual(c1, last_c1)
                last_c1 = c1

                s = str(c1)
                self.assertEqual(s[:20], "NaturalRateCylinder(")
                x = s[20:-1].split(",")
                self.assertEqual(float(x[0]), le)
                self.assertEqual(float(x[1]), r)
                self.assertEqual(float(x[2]), -1)
                self.assertEqual(float(x[3]), 1)

    def check_uniform_pdf(self, c, solid_angle, area_int):
        area_int *= 1e4
        self.assertAlmostEqual(c.etendue, solid_angle * area_int, 4)

        x = np.linspace(c.cos_zen_min, c.cos_zen_max, 10000)
        pdfs = c.pdf(x)
        np.testing.assert_allclose(pdfs, 1 / solid_angle / c.projected_area(x), 1e-15)
        self.assertAlmostEqual((1 / pdfs).sum() / len(x) / c.etendue, 1, 3)

        x = np.linspace(-2, np.nextafter(c.cos_zen_min, -np.inf))
        np.testing.assert_array_equal(c.pdf(x), 0)

        x = np.linspace(np.nextafter(c.cos_zen_max, np.inf), 2)
        np.testing.assert_array_equal(c.pdf(x), 0)

    def test_uniform_solid_angle(self):
        last_c1 = None
        for le in range(100, 1000, 300):
            for r in range(100, 1000, 300):
                c1 = UniformSolidAngleCylinder(le, r, -1, 1)
                self.check_diff_etendue(c1, le, r)
                self.check_uniform_pdf(c1, 4 * np.pi, np.pi / 2 * r * (r + le))

                c2 = UniformSolidAngleCylinder(le, r, -1, 0)
                self.check_uniform_pdf(c2, 2 * np.pi, np.pi / 2 * r * (r + le))

                c3 = UniformSolidAngleCylinder(le, r, 0, 1)
                self.check_uniform_pdf(c3, 2 * np.pi, np.pi / 2 * r * (r + le))

                self.assertEqual(c1, c1)
                self.assertNotEqual(c1, c2)
                self.assertNotEqual(c1, c3)
                self.assertEqual(c2, c2)
                self.assertNotEqual(c2, c3)
                self.assertEqual(c3, c3)
                self.assertNotEqual(c1, last_c1)
                last_c1 = c1

                s = str(c1)
                self.assertEqual(s[:26], "UniformSolidAngleCylinder(")
                x = s[26:-1].split(",")
                self.assertEqual(float(x[0]), le)
                self.assertEqual(float(x[1]), r)
                self.assertEqual(float(x[2]), -1)
                self.assertEqual(float(x[3]), 1)

    def check_circle(self, c, etendue):
        self.assertAlmostEqual(c.etendue, etendue, 4)

        x = np.linspace(c.cos_zen_min, c.cos_zen_max, 10000)
        np.testing.assert_allclose(c.projected_area(x), np.pi * (1e2 * c.radius) ** 2)
        pdfs = c.pdf(x)
        np.testing.assert_allclose(pdfs, 1 / etendue, 1e-15)
        self.assertAlmostEqual((1 / pdfs).sum() / len(x) / c.etendue, 1)

        x = np.linspace(-2, np.nextafter(c.cos_zen_min, -np.inf))
        np.testing.assert_array_equal(c.pdf(x), 0)

        x = np.linspace(np.nextafter(c.cos_zen_max, np.inf), 2)
        np.testing.assert_array_equal(c.pdf(x), 0)

    def test_circle_injector(self):
        last_c1 = None
        for r in range(100, 1000, 300):
            c1 = CircleInjector(r, -1, 1)
            self.check_circle(c1, 4e4 * np.pi**2 * r**2)

            c2 = CircleInjector(r, -1, 0)
            self.check_circle(c2, 2e4 * np.pi**2 * r**2)

            c3 = CircleInjector(r, 0, 1)
            self.check_circle(c3, 2e4 * np.pi**2 * r**2)

            self.assertEqual(c1, c1)
            self.assertNotEqual(c1, c2)
            self.assertNotEqual(c1, c3)
            self.assertEqual(c2, c2)
            self.assertNotEqual(c2, c3)
            self.assertEqual(c3, c3)
            self.assertNotEqual(c1, last_c1)
            last_c1 = c1

            s = str(c1)
            self.assertEqual(s[:15], "CircleInjector(")
            x = s[15:-1].split(",")
            self.assertEqual(float(x[0]), r)
            self.assertEqual(float(x[1]), -1)
            self.assertEqual(float(x[2]), 1)


if __name__ == "__main__":
    unittest.main()
