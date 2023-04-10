#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest
from copy import copy

import numpy as np
from simweights import TIG1996, NaturalRateCylinder, PowerLaw, Weighter, generation_surface


def fluxfun1(energy):
    return energy**2


def fluxfun2(pdgid, energy):  # noqa: ARG001
    return energy**2


def fluxfun3(pdgid, energy, cos_zen):  # noqa: ARG001
    return cos_zen * energy**2


class fake_nuflux:
    def getFlux(self, particle_type, energy, cos_zenith):  # noqa: ARG002
        return energy**-3 / cos_zenith


class TestWeighter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N1 = 15
        cls.data1 = {
            "I3Weight": {
                "type": np.full(cls.N1, 2212),
                "energy": np.linspace(5e5, 5e6, cls.N1),
                "zenith": np.full(cls.N1, np.pi / 4),
            },
        }
        cls.c1 = NaturalRateCylinder(100, 200, 0, 1)
        cls.p1 = PowerLaw(0, 5e5, 5e6)
        cls.s1 = cls.N1 * generation_surface(2212, cls.p1, cls.c1)
        cls.m1 = {
            "pdgid": ("I3Weight", "type"),
            "energy": ("I3Weight", "energy"),
            "zenith": ("I3Weight", "zenith"),
        }
        cls.weighter1 = Weighter([cls.data1], cls.s1)
        for x, y in cls.m1.items():
            cls.weighter1.add_weight_column(x, cls.data1[y[0]][y[1]])
        cls.weighter1.add_weight_column("event_weight", np.full(cls.N1, 1))
        cls.weighter1.add_weight_column("cos_zen", np.cos(cls.data1["I3Weight"]["zenith"]))

        cls.data2 = {
            "weight": {
                "primary_type": np.full(cls.N1, 2212),
                "primary_energy": np.full(cls.N1, 1e6),
                "primary_zenith": np.full(cls.N1, np.pi / 4),
                "ev": np.full(cls.N1, 1),
            },
        }
        cls.m2 = {
            "pdgid": ("weight", "primary_type"),
            "energy": ("weight", "primary_energy"),
            "zenith": ("weight", "primary_zenith"),
            "event_weight": ("weight", "ev"),
        }
        cls.weighter2 = Weighter([cls.data2], cls.s1)
        for x, y in cls.m2.items():
            cls.weighter2.add_weight_column(x, cls.data2[y[0]][y[1]])
        cls.weighter2.add_weight_column("cos_zen", np.cos(cls.data2["weight"]["primary_zenith"]))

    def check_weight(self, weighter, N, val):
        flux = 6
        weights = weighter.get_weights(flux)
        self.assertAlmostEqual(weights.sum(), flux * val, -2)

        flux = list(range(0, N))
        weights = weighter.get_weights(flux)
        np.testing.assert_allclose(weights, np.array(flux) * val / N)

        flux = np.linspace(1, 2, N)
        weights = weighter.get_weights(flux)
        np.testing.assert_allclose(weights, flux * val / N)

        weights = weighter.get_weights(fluxfun1)
        flux_vals = fluxfun1(weighter.get_weight_column("energy"))
        np.testing.assert_allclose(weights, flux_vals * val / N, 1e-2)

        weights = weighter.get_weights(fluxfun2)
        flux_vals = fluxfun2(weighter.get_weight_column("pdgid"), weighter.get_weight_column("energy"))
        np.testing.assert_allclose(weights, flux_vals * val / N)

        weights = weighter.get_weights(fluxfun3)
        flux_vals = fluxfun3(
            weighter.get_weight_column("pdgid"),
            weighter.get_weight_column("energy"),
            np.cos(weighter.get_weight_column("zenith")),
        )
        np.testing.assert_allclose(weights, flux_vals * val / N)

        flux4 = fake_nuflux()
        weights = weighter.get_weights(flux4)
        flux_vals = flux4.getFlux(
            weighter.get_weight_column("pdgid"),
            weighter.get_weight_column("energy"),
            np.cos(weighter.get_weight_column("zenith")),
        )
        np.testing.assert_allclose(weights, flux_vals * val / N)

        flux = TIG1996()
        weights = weighter.get_weights(flux)
        flux_vals = flux(weighter.get_weight_column("energy"), weighter.get_weight_column("pdgid"))
        np.testing.assert_allclose(weights, flux_vals * val / N)

        with self.assertRaises(ValueError):
            self.weighter1.get_weights("flux")

        with self.assertRaises(ValueError):
            self.weighter1.get_weights(None)

    def test_columns(self):
        np.testing.assert_array_equal(
            self.weighter1.get_column("I3Weight", "type"),
            self.weighter1.get_weight_column("pdgid"),
        )
        np.testing.assert_array_equal(
            self.weighter1.get_column("I3Weight", "energy"),
            self.weighter1.get_weight_column("energy"),
        )
        np.testing.assert_array_equal(
            self.weighter1.get_column("I3Weight", "zenith"),
            self.weighter1.get_weight_column("zenith"),
        )
        np.testing.assert_array_equal(
            self.weighter2.get_column("weight", "primary_type"),
            self.weighter2.get_weight_column("pdgid"),
        )
        np.testing.assert_array_equal(
            self.weighter2.get_column("weight", "primary_energy"),
            self.weighter2.get_weight_column("energy"),
        )
        np.testing.assert_array_equal(
            self.weighter2.get_column("weight", "primary_zenith"),
            self.weighter2.get_weight_column("zenith"),
        )
        np.testing.assert_array_equal(
            self.weighter2.get_column("weight", "ev"),
            self.weighter2.get_weight_column("event_weight"),
        )

    def test_weights(self):
        self.check_weight(self.weighter1, self.N1, self.p1.integral * self.c1.etendue)
        self.check_weight(self.weighter2, self.N1, self.p1.integral * self.c1.etendue)

    def test_empty(self):
        fake_file = {"I3Weight": {"energy": [], "type": [], "zenith": []}}
        weighter = Weighter([fake_file], 0)
        weighter.add_weight_column("energy", np.array([]))
        weighter.add_weight_column("pdgid", np.array([]))
        weighter.add_weight_column("cos_zen", np.array([]))
        weights = weighter.get_weights(1)
        self.assertEqual(weights.shape, (0,))

    def test_wrong_size_column(self):
        with self.assertRaises(ValueError):
            self.weighter1.add_weight_column("asdf", np.full(555, 1))

    def test_bad_column(self):
        with self.assertRaises(ValueError):
            self.weighter1.get_weights(lambda asdf: asdf**-2)
        with self.assertRaises(ValueError):
            self.weighter2.get_weights(lambda x: x**-2)

    def test_outside(self):
        data = {
            "I3Weight": {
                "type": np.full(self.N1, 2212),
                "energy": np.full(self.N1, 1e6),
                "zenith": -np.full(self.N1, 3 * np.pi / 4),
            },
        }
        weighter = Weighter([data], self.s1)
        weighter.add_weight_column("pdgid", data["I3Weight"]["type"])
        weighter.add_weight_column("energy", data["I3Weight"]["energy"])
        weighter.add_weight_column("cos_zen", np.cos(data["I3Weight"]["zenith"]))
        weighter.add_weight_column("event_weight", np.full(self.N1, 1))
        with self.assertWarns(UserWarning):
            weights = weighter.get_weights(1)
        np.testing.assert_array_equal(weights, 0)

    def test_effective_area(self):
        self.assertAlmostEqual(
            self.weighter1.effective_area([5e5, 5e6], [0, 1])[0][0],
            self.c1.etendue / 2e4 / np.pi,
            6,
        )
        self.assertAlmostEqual(
            self.weighter1.effective_area([5e5, 5e6], [0, 1], np.ones(self.N1, dtype=bool))[0][0],
            self.c1.etendue / 2e4 / np.pi,
            6,
        )
        np.testing.assert_allclose(
            self.weighter1.effective_area(np.linspace(5e5, 5e6, self.N1 + 1), [0, 1]),
            [np.full(self.N1, self.c1.etendue / 2e4 / np.pi)],
        )

    def test_weighter_addition(self):
        weighter_sum = self.weighter1 + self.weighter1
        w1 = self.weighter1.get_weights(1)
        ws = weighter_sum.get_weights(1)
        self.assertAlmostEqual(w1.sum(), ws.sum(), -2)
        self.assertEqual(2 * len(w1), len(ws))
        self.assertIsNot(self.weighter1, weighter_sum)
        self.assertEqual(2 * len(self.weighter1.data), len(weighter_sum.data))
        self.assertEqual(2 * self.weighter1.surface, weighter_sum.surface)
        self.assertEqual(self.weighter1.colnames, weighter_sum.colnames)
        self.check_weight(weighter_sum, 2 * self.N1, self.p1.integral * self.c1.etendue)

        weightera = copy(self.weighter1)
        weightera += self.weighter1
        self.assertIsNot(self.weighter1, weightera)
        self.assertEqual(2 * len(self.weighter1.data), len(weightera.data))
        self.assertEqual(2 * self.weighter1.surface, weightera.surface)
        self.assertEqual(self.weighter1.colnames, weightera.colnames)
        self.check_weight(weightera, 2 * self.N1, self.p1.integral * self.c1.etendue)

        weighterb = copy(self.weighter1)
        weighterb += self.weighter2
        self.assertEqual(2 * len(self.weighter1.data), len(weighterb.data))
        self.assertEqual(2 * self.weighter1.surface, weighterb.surface)
        self.assertEqual(self.weighter1.colnames, weighterb.colnames)
        self.check_weight(weighterb, 2 * self.N1, self.p1.integral * self.c1.etendue)

        weighterc = self.weighter1 + self.weighter2
        self.assertEqual(2 * len(self.weighter1.data), len(weighterc.data))
        self.assertEqual(2 * self.weighter1.surface, weighterc.surface)
        self.assertEqual(self.weighter1.colnames, weighterc.colnames)
        self.check_weight(weighterc, 2 * self.N1, self.p1.integral * self.c1.etendue)

        weighterd = self.weighter1 + 0
        self.assertEqual(len(self.weighter1.data), len(weighterd.data))
        self.assertEqual(self.weighter1.surface, weighterd.surface)
        self.assertEqual(self.weighter1.colnames, weighterd.colnames)
        self.check_weight(weighterd, self.N1, self.p1.integral * self.c1.etendue)

        weightere = 0 + self.weighter1
        self.assertEqual(len(self.weighter1.data), len(weightere.data))
        self.assertEqual(self.weighter1.surface, weightere.surface)
        self.assertEqual(self.weighter1.colnames, weightere.colnames)
        self.check_weight(weightere, self.N1, self.p1.integral * self.c1.etendue)

        weighterf = sum([self.weighter1, self.weighter2])
        self.assertEqual(2 * len(self.weighter1.data), len(weighterf.data))
        self.assertEqual(2 * self.weighter1.surface, weighterf.surface)
        self.assertEqual(self.weighter1.colnames, weighterf.colnames)
        self.check_weight(weighterf, 2 * self.N1, self.p1.integral * self.c1.etendue)

        with self.assertRaises(TypeError):
            5 + self.weighter1

        with self.assertRaises(TypeError):
            self.weighter1 + 33.3

        with self.assertRaises(TypeError):
            None + self.weighter1

        with self.assertRaises(TypeError):
            self.weighter1 + object()

    def test_nuflux(self):
        try:
            import nuflux
        except ImportError:
            self.skipTest("nuflux not found")

        N1 = 15
        data1 = {
            "I3Weight": {
                "type": np.full(N1, 14, dtype=np.int32),
                "energy": np.linspace(5e5, 5e6, N1),
                "zenith": np.full(N1, np.pi / 4),
            },
        }
        s1 = N1 * generation_surface(14, self.p1, self.c1)
        weighter1 = Weighter([data1], s1)
        weighter1.add_weight_column("pdgid", data1["I3Weight"]["type"])
        weighter1.add_weight_column("energy", data1["I3Weight"]["energy"])
        weighter1.add_weight_column("cos_zen", np.cos(data1["I3Weight"]["zenith"]))
        weighter1.add_weight_column("event_weight", np.full(N1, 1))

        honda = nuflux.makeFlux("honda2006")
        w = weighter1.get_weights(honda)
        fluxval = honda.getFlux(14, data1["I3Weight"]["energy"], np.cos(data1["I3Weight"]["zenith"]))
        oneweight = weighter1.get_weights(1)
        np.testing.assert_allclose(w, fluxval * oneweight)

    def test_string(self):
        string1 = str(self.weighter1)
        self.assertIn(str(self.s1), string1)
        string2 = self.weighter1.tostring(fluxfun1)
        self.assertIn(str(self.s1), string2)
        self.assertIn(f"{self.weighter1.get_weights(fluxfun1).sum():8.6g}", string2)


if __name__ == "__main__":
    unittest.main()
