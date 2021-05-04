#!/usr/bin/env python
import os
import unittest

import numpy as np
import pandas as pd

from simweights import NuGenWeighter


class TestNugenDatasets(unittest.TestCase):
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
        if "SolidAngle" in wd:
            np.testing.assert_allclose(solid_angle, wd["SolidAngle"])

        cylinder = w.surface.spectra[pdgid][0].spatial_dist
        proj_area = cylinder.projected_area(np.cos(wd["PrimaryNeutrinoZenith"]))

        if "InjectionAreaCGS" in wd:
            injection_area = wd["InjectionAreaCGS"]
        if "InjectionAreaNormCGS" in wd:
            injection_area = wd["InjectionAreaNormCGS"]
        np.testing.assert_allclose(proj_area, injection_area / 1e4)

        sw_etendue = 1 / cylinder.pdf(np.cos(wd["PrimaryNeutrinoZenith"]))
        np.testing.assert_allclose(sw_etendue, solid_angle * injection_area / 1e4, 1e-5)

        power_law = w.surface.spectra[pdgid][0].energy_dist
        energy_factor = 1 / power_law.pdf(wd["PrimaryNeutrinoEnergy"])

        if "TotalWeight" in wd:
            total_weight = wd["TotalWeight"]
        elif "TotalInteractionProbabilityWeight" in wd:
            total_weight = wd["TotalInteractionProbabilityWeight"]

        one_weight = total_weight * energy_factor * solid_angle * injection_area
        np.testing.assert_allclose(one_weight, wd["OneWeight"])

        one_weight = (
            total_weight
            / power_law.pdf(wd["PrimaryNeutrinoEnergy"])
            / cylinder.pdf(np.cos(wd["PrimaryNeutrinoZenith"]))
            / 1e-4
        )
        np.testing.assert_allclose(one_weight, wd["OneWeight"], 1e-5)

        if "TypeWeight" in wd:
            type_weight = wd["TypeWeight"]
        else:
            type_weight = 0.5
        np.testing.assert_allclose(
            w.get_weights(1), wd["OneWeight"] / (wd["NEvents"] * type_weight * 1e4), 1e-5
        )

        f.close()

    def test_20885(self):
        self.cmp_dataset("Level2_IC86.2016_NuE.020885.000000.hdf5")

    def test_20878(self):
        self.cmp_dataset("Level2_IC86.2016_NuMu.020878.000000.hdf5")

    def test_20895(self):
        self.cmp_dataset("Level2_IC86.2016_NuTau.020895.000000.hdf5")

    def test_12646(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nue.012646.000000.clsim-base-4.0.5.0.99_eff.hdf5")

    def test_12034(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nue.012034.000000.clsim-base-4.0.3.0.99_eff.hdf5")

    def test_11981(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nue.011981.000000.clsim-base-4.0.3.0.99_eff.hdf5")

    def test_11883(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_numu.011883.000000.clsim-base-4.0.5.0.99_eff.hdf5")

    def test_11836(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nutau.011836.000000.clsim-base-4.0.3.0.99_eff.hdf5")

    def test_11477(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nutau.011477.000000.clsim-base-4.0.3.0.99_eff.hdf5")

    def test_11374(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.hdf5")

    def test_11297(self):
        self.cmp_dataset("Level2_nugen_nutau_IC86.2012.011297.000000.hdf5")

    def test_11070(self):
        self.cmp_dataset("Level2_nugen_numu_IC86.2012.011070.000000.hdf5")

    def test_11069(self):
        self.cmp_dataset("Level2_nugen_numu_IC86.2012.011069.000000.hdf5")

    def test_11065(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_NuTau.011065.000001.hdf5")

    def test_11029(self):
        self.cmp_dataset("Level2_nugen_numu_IC86.2012.011029.000000.hdf5")

    def test_20692(self):
        self.cmp_dataset("Level2_IC86.2011_nugen_NuE.010692.000000.hdf5")

    def test_10634(self):
        self.cmp_dataset("Level2_IC86.2011_nugen_NuMu.010634.000000.hdf5")


if __name__ == "__main__":
    unittest.main()
