#!/usr/bin/env python
import os
import unittest

import numpy as np
import pandas as pd

from simweights import NuGenWeighter


def unit_flux(energy, pdgid, cos_zen):
    return 1


class TestCorsikaDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        datadir = os.environ.get("SIMWEIGHTS_DATA", None)
        if not datadir:
            cls.skipTest(None, "enviornment varible SIMWEIGHTS_DATA not set")
        cls.datadir = datadir + "/"

    def cmp_dataset(self, fname):
        f = pd.HDFStore(self.datadir + "/" + fname, "r")
        wd = f["I3MCWeightDict"]
        w = NuGenWeighter(f, nfiles=1)
        pdgid = wd["PrimaryNeutrinoType"][0]

        solid_angle = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))
        np.testing.assert_allclose(solid_angle, wd["SolidAngle"])

        cylinder = w.surface.spectra[pdgid][0].spatial_dist
        proj_area = cylinder.projected_area(np.cos(wd["PrimaryNeutrinoZenith"]))
        np.testing.assert_allclose(proj_area, wd["InjectionAreaCGS"] / 1e4)

        sw_etendue = 1 / cylinder.pdf(np.cos(wd["PrimaryNeutrinoZenith"]))
        np.testing.assert_allclose(sw_etendue, wd["SolidAngle"] * wd["InjectionAreaCGS"] / 1e4, 1e-5)

        power_law = w.surface.spectra[pdgid][0].energy_dist
        energy_factor = 1 / power_law.pdf(wd["PrimaryNeutrinoEnergy"])
        one_weight = wd["TotalWeight"] * energy_factor * wd["InjectionAreaCGS"] * wd["SolidAngle"]
        np.testing.assert_allclose(one_weight, wd["OneWeight"])

        one_weight = wd["TotalWeight"] / (
            power_law.pdf(wd["PrimaryNeutrinoEnergy"])
            * cylinder.pdf(np.cos(wd["PrimaryNeutrinoZenith"]))
            * 1e-4
        )
        np.testing.assert_allclose(one_weight, wd["OneWeight"], 1e-5)

        np.testing.assert_allclose(
            w.get_weights(unit_flux), wd["OneWeight"] / (wd["NEvents"] * wd["TypeWeight"] * 1e4), 1e-5
        )

    def test_20885(self):
        self.cmp_dataset("Level2_IC86.2016_NuE.020885.000000.hdf5")

    def test_20878(self):
        self.cmp_dataset("Level2_IC86.2016_NuMu.020878.000000.hdf5")

    def test_20895(self):
        self.cmp_dataset("Level2_IC86.2016_NuTau.020895.000000.hdf5")

    # def test_(self):
    #     self.cmp_dataset("Level2_IC86.2015_corsika.0.000000.hdf5")


if __name__ == "__main__":
    unittest.main()
