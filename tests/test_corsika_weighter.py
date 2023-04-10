#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
import simweights
from scipy.interpolate import interp1d
from simweights import CorsikaWeighter

info_dtype = [
    ("n_events", np.int32),
    ("primary_type", np.int32),
    ("oversampling", np.int32),
    ("cylinder_height", np.float64),
    ("cylinder_radius", np.float64),
    ("min_zenith", np.float64),
    ("max_zenith", np.float64),
    ("min_energy", np.float64),
    ("max_energy", np.float64),
    ("power_law_index", np.float64),
]

primary_dtype = [("type", np.int32), ("energy", np.float64), ("zenith", np.float64)]

weight_dtype = [
    ("ParticleType", np.int32),
    ("CylinderLength", np.float64),
    ("CylinderRadius", np.float64),
    ("ThetaMin", np.float64),
    ("ThetaMax", np.float64),
    ("OverSampling", np.float64),
    ("Weight", np.float64),
    ("NEvents", np.float64),
    ("EnergyPrimaryMin", np.float64),
    ("EnergyPrimaryMax", np.float64),
    ("PrimarySpectralIndex", np.float64),
]


def get_cos_zenith_dist(c, N):
    cz = np.linspace(c.cos_zen_min, c.cos_zen_max, 1000)
    cdf = (c._diff_etendue(cz) - c._diff_etendue(c.cos_zen_min)) / (
        c._diff_etendue(c.cos_zen_max) - c._diff_etendue(c.cos_zen_min)
    )
    cf = interp1d(cdf, cz)
    p = np.linspace(0, 1, N)
    return cf(p)


def make_corsika_data(pdgid, nevents, c, p):
    weight = np.zeros(nevents, dtype=weight_dtype)
    weight["ParticleType"] = pdgid
    weight["CylinderLength"] = c.length
    weight["CylinderRadius"] = c.radius
    weight["ThetaMin"] = np.arccos(c.cos_zen_max)
    weight["ThetaMax"] = np.arccos(c.cos_zen_min)
    weight["OverSampling"] = 1
    weight["Weight"] = 1
    weight["NEvents"] = nevents
    weight["EnergyPrimaryMin"] = p.a
    weight["EnergyPrimaryMax"] = p.b
    weight["PrimarySpectralIndex"] = p.g

    primary = np.zeros(nevents, dtype=primary_dtype)
    primary["type"] = pdgid
    primary["zenith"] = np.arccos(get_cos_zenith_dist(c, nevents))
    np.random.default_rng().shuffle(primary["zenith"])
    primary["energy"] = p.ppf(np.linspace(0, 1, nevents))
    return {"CorsikaWeightMap": weight, "PolyplopiaPrimary": primary}


class TestCorsikaWeighter(unittest.TestCase):
    def test_old_corsika(self):
        c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
        p1 = simweights.PowerLaw(0, 1e3, 1e4)
        d = make_corsika_data(2212, 10000, c1, p1)

        for oversampling in [1, 5, 50]:
            d["CorsikaWeightMap"]["OverSampling"] = oversampling
            for nfiles in [1, 10, 100]:
                wobj = CorsikaWeighter(d, nfiles=nfiles)

                for flux in [0.1, 1, 10]:
                    w = wobj.get_weights(flux)
                    np.testing.assert_allclose(
                        w.sum(),
                        flux * c1.etendue * p1.integral / nfiles / oversampling,
                    )
                    E = d["PolyplopiaPrimary"]["energy"]
                    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / nfiles / oversampling, 5e-3)

        with self.assertRaises(RuntimeError):
            CorsikaWeighter(d)

        with self.assertRaises(TypeError):
            CorsikaWeighter(d, nfiles=object())

    def test_sframe_corsika(self):
        c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
        p1 = simweights.PowerLaw(0, 1e3, 1e4)
        d = make_corsika_data(2212, 10000, c1, p1)
        for oversampling in [1, 5, 50]:
            for nfiles in [1, 10, 100]:
                rows = nfiles * [
                    (
                        10000,
                        2212,
                        oversampling,
                        c1.length,
                        c1.radius,
                        np.arccos(c1.cos_zen_max),
                        np.arccos(c1.cos_zen_min),
                        p1.a,
                        p1.b,
                        p1.g,
                    ),
                ]
                d["I3CorsikaInfo"] = np.array(rows, dtype=info_dtype)
                wobj = CorsikaWeighter(d)

                for flux in [0.1, 1, 10]:
                    w = wobj.get_weights(flux)
                    np.testing.assert_allclose(
                        w.sum(),
                        flux * c1.etendue * p1.integral / nfiles / oversampling,
                    )
                    E = d["PolyplopiaPrimary"]["energy"]
                    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / nfiles / oversampling, 5e-3)

        with self.assertWarns(UserWarning):
            CorsikaWeighter(d, nfiles=10)

    def test_triggered_corsika(self):
        weight_dtype = [
            ("type", np.int32),
            ("energy", np.float64),
            ("zenith", np.float64),
            ("weight", np.float64),
        ]
        info_dtype = [
            ("primary_type", np.int32),
            ("n_events", np.int32),
            ("cylinder_height", np.float64),
            ("cylinder_radius", np.float64),
            ("min_zenith", np.float64),
            ("max_zenith", np.float64),
            ("min_energy", np.float64),
            ("max_energy", np.float64),
            ("power_law_index", np.float64),
        ]

        nevents = 10000
        c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
        p1 = simweights.PowerLaw(0, 1e3, 1e4)
        d = make_corsika_data(2212, 10000, c1, p1)
        weight = np.zeros(nevents, dtype=weight_dtype)
        weight["type"] = 2212
        weight["zenith"] = np.arccos(get_cos_zenith_dist(c1, nevents))
        weight["energy"] = p1.ppf(np.linspace(0, 1, nevents))

        for event_weight in [1e-6, 1e-3, 1]:
            weight["weight"] = event_weight

            for nfiles in [1, 5, 50]:
                rows = nfiles * [
                    (
                        2212,
                        nevents,
                        c1.length,
                        c1.radius,
                        np.arccos(c1.cos_zen_max),
                        np.arccos(c1.cos_zen_min),
                        p1.a,
                        p1.b,
                        p1.g,
                    ),
                ]
                info = np.array(rows, dtype=info_dtype)
                d = {"I3CorsikaWeight": weight, "I3PrimaryInjectorInfo": info}

                for flux in [0.1, 1, 10]:
                    wobj = CorsikaWeighter(d)
                    w = wobj.get_weights(flux)
                    np.testing.assert_allclose(
                        w.sum(),
                        flux * event_weight * c1.etendue * p1.integral / nfiles,
                    )
                    E = d["I3CorsikaWeight"]["energy"]
                    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, flux * event_weight * Ewidth * c1.etendue / nfiles, 5e-3)

        with self.assertRaises(RuntimeError):
            CorsikaWeighter(d, nfiles=10)

        with self.assertRaises(RuntimeError):
            CorsikaWeighter({"I3CorsikaWeight": weight})


if __name__ == "__main__":
    unittest.main()
