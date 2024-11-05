#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
import pandas as pd

from simweights import CircleInjector, GenieWeighter, PowerLaw

mcwd_keys = [
    "NEvents",
    "MinZenith",
    "MaxZenith",
    "PowerLawIndex",
    "MinEnergyLog",
    "MaxEnergyLog",
    "InjectionSurfaceR",
    "PrimaryNeutrinoEnergy",
    "GeneratorVolume",
]

grd_keys = ["neu", "pxv", "pyv", "pzv", "Ev", "wght", "_glbprbscale"]


def make_new_table(pdgid, nevents, spatial, spectrum, coszen):
    dtype = [(k, float) for k in mcwd_keys]
    weight = np.zeros(nevents, dtype=dtype)
    weight["NEvents"] = nevents
    weight["MinZenith"] = np.arccos(spatial.cos_zen_max)
    weight["MaxZenith"] = np.arccos(spatial.cos_zen_min)
    weight["PowerLawIndex"] = -1 * spectrum.g
    weight["MinEnergyLog"] = np.log10(spectrum.a)
    weight["MaxEnergyLog"] = np.log10(spectrum.b)
    weight["InjectionSurfaceR"] = spatial.radius
    weight["GeneratorVolume"] = 1.0
    weight["PrimaryNeutrinoEnergy"] = spectrum.ppf(np.linspace(0, 1, nevents, endpoint=False))

    dtype = [(k, float) for k in grd_keys]
    resultdict = np.zeros(nevents, dtype=dtype)
    resultdict["neu"] = pdgid
    resultdict["pxv"] = 1
    resultdict["pyv"] = 1
    resultdict["pzv"] = -coszen
    resultdict["Ev"] = weight["PrimaryNeutrinoEnergy"]
    resultdict["wght"] = 1.0
    resultdict["_glbprbscale"] = 1.0

    return weight, resultdict


class TestGenieIcetrayWeighter(unittest.TestCase):
    def test_genie_icetray(self):
        nevents = 100000
        coszen = 0.7
        pdgid = 12
        c1 = CircleInjector(300, 0, 1)
        p1 = PowerLaw(0, 1e3, 1e4)

        t1 = make_new_table(pdgid, nevents, c1, p1, coszen)

        mcwd = pd.DataFrame(t1[0])
        grd = pd.DataFrame(t1[1])

        f1 = {"I3MCWeightDict": mcwd, "I3GENIEResultDict": grd}

        for nfiles in [1, 10, 100]:
            wf = GenieWeighter(f1, nfiles=nfiles)
            for flux in [1e-6, 1, 1e6]:
                w1 = wf.get_weights(flux)
                w2 = flux * p1.integral * c1.etendue / (0.7 * nfiles)
                np.testing.assert_allclose(w1.sum(), w2)
                E = mcwd["PrimaryNeutrinoEnergy"]
                y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
                Ewidth = np.ediff1d(x)
                np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / (0.7 * nfiles), 6e-3)

    def test_empty(self):
        with self.assertRaises(RuntimeError):
            x = {"I3MCWeightDict": {key: [] for key in mcwd_keys}, "I3GENIEResultDict": {key: [] for key in grd_keys}}
            GenieWeighter(x, nfiles=1)

        with self.assertRaises(RuntimeError):
            x = {"I3MCWeightDict": {key: [1] for key in mcwd_keys}, "I3GENIEResultDict": {key: [1] for key in grd_keys}}
            GenieWeighter(x)


if __name__ == "__main__":
    unittest.main()
