#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np
import pandas as pd
from simweights import CircleInjector, NaturalRateCylinder, NuGenWeighter, PowerLaw

base_keys = [
    "NEvents",
    "MinZenith",
    "MaxZenith",
    "PowerLawIndex",
    "MinEnergyLog",
    "MaxEnergyLog",
    "PrimaryNeutrinoType",
    "PrimaryNeutrinoZenith",
    "PrimaryNeutrinoEnergy",
]


def make_new_table(pdgid, nevents, spatial, spectrum):
    dtype = [(k, float) for k in base_keys]
    weight = np.zeros(nevents, dtype=dtype)
    weight["PrimaryNeutrinoType"] = pdgid
    weight["NEvents"] = nevents
    weight["MinZenith"] = np.arccos(spatial.cos_zen_max)
    weight["MaxZenith"] = np.arccos(spatial.cos_zen_min)
    weight["PowerLawIndex"] = spectrum.g
    weight["MinEnergyLog"] = np.log10(spectrum.a)
    weight["MaxEnergyLog"] = np.log10(spectrum.b)
    weight["PrimaryNeutrinoZenith"] = np.arccos(
        np.linspace(spatial.cos_zen_max, spatial.cos_zen_min, nevents),
    )
    weight["PrimaryNeutrinoEnergy"] = spectrum.ppf(np.linspace(0, 1, nevents))
    return weight


class TestNugenWeighter(unittest.TestCase):
    def test_nugen_energy_post_V6(self):
        p1 = PowerLaw(0, 1e3, 1e4)
        c1 = NaturalRateCylinder(200, 100, 0, 0.001)
        t1 = pd.DataFrame(make_new_table(14, 10000, c1, p1))
        t1["CylinderHeight"] = c1.length
        t1["CylinderRadius"] = c1.radius
        t1["TypeWeight"] = 0.5

        for weight in 0.1, 1, 10:
            t1["TotalWeight"] = weight
            f1 = {"I3MCWeightDict": t1}
            for nfiles in [1, 10, 100]:
                wf = NuGenWeighter(f1, nfiles=nfiles)
                for flux in [1e-6, 1, 1e6]:
                    w1 = wf.get_weights(flux)
                    np.testing.assert_allclose(
                        w1.sum(),
                        2 * weight * flux * p1.integral * c1.etendue / nfiles,
                    )
                    E = t1["PrimaryNeutrinoEnergy"]
                    y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, 2 * weight * flux * Ewidth * c1.etendue / nfiles, 6e-3)

    def test_nugen_energy_pre_V6(self):
        p1 = PowerLaw(0, 1e3, 1e4)
        c1 = NaturalRateCylinder(1900, 950, 0, 0.001)
        t1 = pd.DataFrame(make_new_table(14, 10000, c1, p1))
        for weight in 0.1, 1, 10:
            t1["TotalWeight"] = weight
            t1["InjectionSurfaceR"] = -1
            t1["TypeWeight"] = 0.5
            f1 = {"I3MCWeightDict": t1}
            for nfiles in [1, 10, 100]:
                wf = NuGenWeighter(f1, nfiles=nfiles)
                for flux in [1e-6, 1, 1e6]:
                    w1 = wf.get_weights(flux)
                    np.testing.assert_allclose(
                        w1.sum(),
                        2 * weight * flux * p1.integral * c1.etendue / nfiles,
                    )
                    E = t1["PrimaryNeutrinoEnergy"]
                    y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, 2 * weight * flux * Ewidth * c1.etendue / nfiles, 6e-3)

    def test_nugen_energy_pre_V04_00(self):
        p1 = PowerLaw(0, 1e3, 1e4)
        c1 = CircleInjector(500, 0, 0.001)
        t1 = pd.DataFrame(make_new_table(14, 10000, c1, p1))
        for weight in 0.1, 1, 10:
            t1["TotalInteractionProbabilityWeight"] = weight
            t1["InjectionSurfaceR"] = c1.radius
            f1 = {"I3MCWeightDict": t1}
            for nfiles in [1, 10, 100]:
                wf = NuGenWeighter(f1, nfiles=nfiles)
                for flux in [1e-6, 1, 1e6]:
                    w1 = wf.get_weights(flux)
                    np.testing.assert_allclose(
                        w1.sum(),
                        2 * weight * flux * p1.integral * c1.etendue / nfiles,
                    )
                    E = t1["PrimaryNeutrinoEnergy"]
                    y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
                    Ewidth = np.ediff1d(x)
                    np.testing.assert_allclose(y, 2 * weight * flux * Ewidth * c1.etendue / nfiles, 6e-3)


if __name__ == "__main__":
    unittest.main()
