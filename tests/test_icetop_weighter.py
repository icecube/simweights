#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
import simweights

info_dtype = [
    ("primary_type", np.int32),
    ("n_events", np.int32),
    ("sampling_radius", np.float64),
    ("min_zenith", np.float64),
    ("max_zenith", np.float64),
    ("min_energy", np.float64),
    ("max_energy", np.float64),
    ("power_law_index", np.float64),
]

particle_dtype = [("type", np.int32), ("energy", np.float64), ("zenith", np.float64)]


class TestIceTopWeighter(unittest.TestCase):
    def test_icetop_corsika(self):
        nevents = 10000
        pdgid = 12
        c1 = simweights.CircleInjector(300, 0, 1)
        p1 = simweights.PowerLaw(0, 1e3, 1e4)

        weight = np.zeros(nevents, dtype=particle_dtype)
        weight["type"] = pdgid
        weight["energy"] = p1.ppf(np.linspace(0, 1, nevents))
        weight["zenith"] = np.arccos(np.linspace(c1.cos_zen_min, c1.cos_zen_max, nevents))

        for nfiles in [1, 5, 50]:
            rows = nfiles * [
                (
                    pdgid,
                    nevents,
                    c1.radius,
                    np.arccos(c1.cos_zen_max),
                    np.arccos(c1.cos_zen_min),
                    p1.a,
                    p1.b,
                    p1.g,
                ),
            ]
            info = np.array(rows, dtype=info_dtype)
            d = {"MCPrimary": weight, "I3TopInjectorInfo": info}

            for flux in [0.1, 1, 10]:
                wobj = simweights.IceTopWeighter(d)
                w = wobj.get_weights(flux)
                np.testing.assert_allclose(w.sum(), flux * c1.etendue * p1.integral / nfiles)
                E = d["MCPrimary"]["energy"]
                y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
                Ewidth = np.ediff1d(x)
                np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / nfiles, 5e-3)

        with self.assertRaises(TypeError):
            simweights.IceTopWeighter(d, nfiles=10)

        with self.assertRaises(KeyError):
            simweights.IceTopWeighter({"MCParticle": weight})

        with self.assertRaises(KeyError):
            simweights.IceTopWeighter({"I3TopInjectorInfo": info})


if __name__ == "__main__":
    unittest.main()
