#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import os
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tables
import uproot
from simweights import NuGenWeighter


class TestNugenDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
        if not datadir:
            cls.skipTest(None, "environment variable SIMWEIGHTS_TESTDATA not set")
        cls.datadir = datadir + "/"

    def cmp_dataset(self, fname):
        filename = Path(self.datadir) / fname
        reffile = h5py.File(str(filename) + ".hdf5", "r")

        wd = reffile["I3MCWeightDict"]
        pdgid = wd["PrimaryNeutrinoType"][0]

        solid_angle = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))
        if "SolidAngle" in wd.dtype.names:
            np.testing.assert_allclose(solid_angle, wd["SolidAngle"])

        if "InjectionAreaCGS" in wd.dtype.names:
            injection_area = wd["InjectionAreaCGS"]
        if "InjectionAreaNormCGS" in wd.dtype.names:
            injection_area = wd["InjectionAreaNormCGS"]

        if "TotalWeight" in wd.dtype.names:
            total_weight = wd["TotalWeight"]
        elif "TotalInteractionProbabilityWeight" in wd.dtype.names:
            total_weight = wd["TotalInteractionProbabilityWeight"]

        type_weight = wd["TypeWeight"] if "TypeWeight" in wd.dtype.names else 0.5
        w0 = wd["OneWeight"] / (wd["NEvents"] * type_weight)

        fobjs = [
            reffile,
            uproot.open(str(filename) + ".root"),
            tables.open_file(str(filename) + ".hdf5", "r"),
            pd.HDFStore(str(filename) + ".hdf5", "r"),
        ]

        for fobj in fobjs:
            with self.subTest(lib=str(fobj)):
                w = NuGenWeighter(fobj, nfiles=1)

                event_weight = w.get_weight_column("event_weight")
                np.testing.assert_allclose(event_weight, total_weight)

                cylinder = w.surface.spectra[pdgid][0].spatial_dist
                proj_area = cylinder.projected_area(w.get_weight_column("cos_zen"))
                np.testing.assert_allclose(proj_area, injection_area)

                sw_etendue = 1 / cylinder.pdf(w.get_weight_column("cos_zen"))
                np.testing.assert_allclose(sw_etendue, solid_angle * injection_area, 1e-5)

                power_law = w.surface.spectra[pdgid][0].energy_dist
                energy_factor = 1 / power_law.pdf(w.get_weight_column("energy"))
                one_weight = (
                    w.get_weight_column("event_weight") * energy_factor * solid_angle * injection_area
                )
                np.testing.assert_allclose(one_weight, wd["OneWeight"])

                np.testing.assert_allclose(w.get_weights(1), w0, 1e-5)

        for fobj in fobjs:
            fobj.close()

    def test_20885(self):
        self.cmp_dataset("Level2_IC86.2016_NuE.020885.000000")

    def test_20878(self):
        self.cmp_dataset("Level2_IC86.2016_NuMu.020878.000000")

    def test_20895(self):
        self.cmp_dataset("Level2_IC86.2016_NuTau.020895.000000")

    def test_12646(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nue.012646.000000.clsim-base-4.0.5.0.99_eff")

    def test_11836(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nutau.011836.000000.clsim-base-4.0.3.0.99_eff")

    def test_11477(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_nutau.011477.000000.clsim-base-4.0.3.0.99_eff")

    def test_11374(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff")

    def test_11297(self):
        self.cmp_dataset("Level2_nugen_nutau_IC86.2012.011297.000000")

    def test_11070(self):
        self.cmp_dataset("Level2_nugen_numu_IC86.2012.011070.000000")

    def test_11069(self):
        self.cmp_dataset("Level2_nugen_numu_IC86.2012.011069.000000")

    def test_11065(self):
        self.cmp_dataset("Level2_IC86.2012_nugen_NuTau.011065.000001")

    def test_11029(self):
        self.cmp_dataset("Level2_nugen_numu_IC86.2012.011029.000000")

    def test_20692(self):
        self.cmp_dataset("Level2_IC86.2011_nugen_NuE.010692.000000")

    def test_10634(self):
        self.cmp_dataset("Level2_IC86.2011_nugen_NuMu.010634.000000")


if __name__ == "__main__":
    unittest.main()
