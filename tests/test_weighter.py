#!/usr/bin/env python

import unittest
from copy import copy

import numpy as np

from simweights import TIG1996, GenerationSurface, NaturalRateCylinder, PowerLaw
from simweights.weighter import Weighter


def fluxfun1(energy):
    print("FLUXFUN", energy)
    return energy ** 2


def fluxfun2(pdgid, energy):
    print("FLUXFUN", energy)
    return energy ** 2


def fluxfun3(pdgid, energy, cos_zen):
    print("FLUXFUN", energy)
    return cos_zen * energy ** 2


class TestWeighter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.N1 = 15
        cls.data1 = dict(
            weight=dict(
                pdgid=np.full(cls.N1, 2212),
                energy=np.linspace(5e5, 5e6, cls.N1),  # np.full(N,1e6),
                zenith=np.full(cls.N1, np.pi / 4),
            )
        )
        cls.c1 = NaturalRateCylinder(100, 200, 0, 1)
        cls.p1 = PowerLaw(0, 5e5, 5e6)
        cls.s1 = cls.N1 * GenerationSurface(2212, cls.p1, cls.c1)
        cls.m1 = dict(
            pdgid=("weight", "pdgid"),
            energy=("weight", "energy"),
            zenith=("weight", "zenith"),
            event_weight=None,
        )
        cls.weighter1 = Weighter([cls.data1], cls.s1, cls.m1)

        cls.data2 = dict(
            weight=dict(
                pdgid=np.full(cls.N1, 2212),
                energy=np.full(cls.N1, 1e6),
                zenith=np.full(cls.N1, np.pi / 4),
                ev=np.full(cls.N1, 8),
            )
        )
        cls.m2 = dict(
            pdgid=("weight", "pdgid"),
            energy=("weight", "energy"),
            zenith=("weight", "zenith"),
            event_weight=("weight", "ev"),
        )
        cls.weighter2 = Weighter([cls.data2], cls.s1, cls.m2)

    def check_weight(self, weighter, N, data, val):
        flux = 6
        weights = weighter.get_weights(flux)
        self.assertAlmostEqual(weights.sum(), flux * val, 1)

        flux = list(range(0, N))
        weights = weighter.get_weights(flux)
        np.testing.assert_allclose(weights, np.array(flux) * val / N)

        flux = np.linspace(1, 2, N)
        weights = weighter.get_weights(flux)
        np.testing.assert_allclose(weights, flux * val / N)

        weights = weighter.get_weights(fluxfun1)
        flux_vals = fluxfun1(data["weight"]["energy"])
        np.testing.assert_allclose(weights, flux_vals * val / N)

        weights = weighter.get_weights(fluxfun2)
        flux_vals = fluxfun2(data["weight"]["pdgid"], data["weight"]["energy"])
        np.testing.assert_allclose(weights, flux_vals * val / N)

        weights = weighter.get_weights(fluxfun3)
        flux_vals = fluxfun3(
            data["weight"]["pdgid"], data["weight"]["energy"], np.cos(data["weight"]["zenith"])
        )
        np.testing.assert_allclose(weights, flux_vals * val / N)

        flux = TIG1996()
        weights = weighter.get_weights(flux)
        flux_vals = flux(data["weight"]["energy"], data["weight"]["pdgid"])
        np.testing.assert_allclose(weights, flux_vals * val / N)

        with self.assertRaises(ValueError):
            self.weighter1.get_weights("flux")

        with self.assertRaises(ValueError):
            self.weighter1.get_weights(None)

    def test_weights(self):
        self.check_weight(self.weighter1, self.N1, self.data1, self.p1.integral * self.c1.etendue)
        self.check_weight(self.weighter2, self.N1, self.data2, 8 * self.p1.integral * self.c1.etendue)

    def test_outside(self):
        data = dict(
            weight=dict(
                pdgid=np.full(self.N1, 2212),
                energy=np.full(self.N1, 1e6),
                zenith=-np.full(self.N1, 3 * np.pi / 4),
            )
        )
        weighter = Weighter([data], self.s1, self.m1)
        with self.assertWarns(UserWarning):
            weights = weighter.get_weights(1)
        np.testing.assert_array_equal(weights, 0)

    def test_effective_area(self):
        self.assertAlmostEqual(self.weighter1.effective_area(2212)[0][0], self.c1.etendue / 2 / np.pi)
        self.assertAlmostEqual(self.weighter1.effective_area()[0][0], self.c1.etendue / 2 / np.pi)
        np.testing.assert_allclose(
            self.weighter1.effective_area(2212, np.linspace(5e5, 5e6, self.N1 + 1)),
            [np.full(self.N1, self.c1.etendue / 2 / np.pi)],
        )

    def test_weighter_addition(self):

        weighter_sum = self.weighter1 + self.weighter1
        w1 = self.weighter1.get_weights(1)
        ws = weighter_sum.get_weights(1)
        self.assertAlmostEqual(w1.sum(), ws.sum())
        self.assertEqual(2 * len(w1), len(ws))
        self.assertIsNot(self.weighter1, weighter_sum)
        self.assertEqual(2 * len(self.weighter1.data), len(weighter_sum.data))
        self.assertEqual(2 * self.weighter1.surface, weighter_sum.surface)
        self.assertEqual(self.weighter1.event_map, weighter_sum.event_map)

        weightera = copy(self.weighter1)
        holder = weightera
        weightera += self.weighter1

        self.assertAlmostEqual(w1.sum(), ws.sum())
        self.assertEqual(2 * len(w1), len(ws))
        self.assertIs(holder, weightera)
        self.assertIsNot(self.weighter1, weightera)
        self.assertEqual(2 * len(self.weighter1.data), len(weightera.data))
        self.assertEqual(2 * self.weighter1.surface, weightera.surface)
        self.assertEqual(self.weighter1.event_map, weightera.event_map)

        weighterb = copy(self.weighter1)
        with self.assertRaises(ValueError):
            weighterb += self.weighter2

        with self.assertRaises(ValueError):
            self.weighter1 + self.weighter2

    def test_nuflux(self):
        try:
            import nuflux
        except ImportError:
            self.skipTest("nuflux not found")

        N1 = 15
        data1 = dict(
            weight=dict(
                pdgid=np.full(N1, 14),
                energy=np.linspace(5e5, 5e6, N1),
                zenith=np.full(N1, np.pi / 4),
            )
        )
        s1 = N1 * GenerationSurface(14, self.p1, self.c1)
        weighter1 = Weighter([data1], s1, self.m1)

        honda = nuflux.makeFlux("honda2006")
        w = weighter1.get_weights(honda)
        fluxval = 1e4 * honda.getFlux(14, data1["weight"]["energy"], np.cos(data1["weight"]["zenith"]))
        oneweight = weighter1.get_weights(1)
        np.testing.assert_allclose(w, fluxval * oneweight)


if __name__ == "__main__":
    unittest.main()
