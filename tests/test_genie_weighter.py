#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
import simweights

info_dtype = [
    ("primary_type", np.int32),
    ("n_flux_events", np.int32),
    ("global_probability_scale", np.float64),
    ("cylinder_radius", np.float64),
    ("min_zenith", np.float64),
    ("max_zenith", np.float64),
    ("min_energy", np.float64),
    ("max_energy", np.float64),
    ("power_law_index", np.float64),
]

result_dtype = [("neu", np.int32), ("Ev", np.float64), ("wght", np.float64)]


class TestCorsikaWeighter(unittest.TestCase):
    def test_triggered_corsika(self):
        nevents = 10000
        pdgid = 12
        c1 = simweights.CircleInjector(300, 0, 1)
        p1 = simweights.PowerLaw(0, 1e3, 1e4)

        weight = np.zeros(nevents, dtype=result_dtype)
        weight["neu"] = pdgid
        weight["Ev"] = p1.ppf(np.linspace(0, 1, nevents))

        for event_weight in [1e-6, 1e-3, 1]:
            weight["wght"] = event_weight

            for nfiles in [1, 5, 50]:
                rows = nfiles * [
                    (
                        pdgid,
                        nevents,
                        1,
                        c1.radius,
                        np.arccos(c1.cos_zen_max),
                        np.arccos(c1.cos_zen_min),
                        p1.a,
                        p1.b,
                        p1.g,
                    ),
                ]
                info = np.array(rows, dtype=info_dtype)
                d = {"I3GenieResult": weight, "I3GenieInfo": info}

                for flux in [0.1, 1, 10]:
                    wobj = simweights.GenieWeighter(d)
                    w = wobj.get_weights(flux)
                    np.testing.assert_allclose(
                        w.sum(),
                        flux * event_weight * c1.etendue * p1.integral / nfiles,
                    )
                    E = d["I3GenieResult"]["Ev"]
                    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, flux * event_weight * Ewidth * c1.etendue / nfiles, 5e-3)

        with self.assertRaises(TypeError):
            simweights.GenieWeighter(d, nfiles=10)

        with self.assertRaises(KeyError):
            simweights.GenieWeighter({"I3CorsikaWeight": weight})


if __name__ == "__main__":
    unittest.main()
