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
from simweights import GenieWeighter
from simweights._utils import get_column, get_table


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

        solid_angle = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))
        injection_area = np.pi * (wd["InjectionSurfaceR"] * 1e2) ** 2
        total_prob = wd["TotalInteractionProbabilityWeight"]

        pli = -wd["PowerLawIndex"][0]
        energy_integral = (
            (10 ** wd["MaxEnergyLog"][0]) ** (pli + 1) - (10 ** wd["MinEnergyLog"][0]) ** (pli + 1)
        ) / (pli + 1)
        energy_factor = 1 / (wd["PrimaryNeutrinoEnergy"] ** pli / energy_integral)
        one_weight = total_prob * energy_factor * solid_angle * injection_area
        np.testing.assert_allclose(one_weight, wd["OneWeight"])
        final_weight = wd["OneWeight"] / (get_column(get_table(reffile, "I3GenieInfo"), "n_flux_events")[0])

        fobjs = [
            reffile,
            uproot.open(str(filename) + ".root"),
            tables.open_file(str(filename) + ".hdf5", "r"),
            pd.HDFStore(str(filename) + ".hdf5", "r"),
        ]

        for fobj in fobjs:
            with self.subTest(lib=str(fobj)):
                w = GenieWeighter(fobj)

                np.testing.assert_allclose(w.get_weight_column("event_weight"), total_prob)

                cylinder = w.surface.spectra[14][0].spatial_dist
                proj_area = cylinder.projected_area(0)
                np.testing.assert_allclose(proj_area, injection_area)

                sw_etendue = 1 / cylinder.pdf(0)
                np.testing.assert_allclose(sw_etendue, solid_angle * injection_area, 1e-5)

                power_law = w.surface.spectra[14][0].energy_dist
                energy_term = 1 / power_law.pdf(w.get_weight_column("energy"))
                np.testing.assert_allclose(energy_term, energy_factor)

                one_weight = w.get_weight_column("event_weight") * energy_term / cylinder.pdf(0)
                np.testing.assert_allclose(one_weight, wd["OneWeight"], 1e-5)

                np.testing.assert_allclose(w.get_weights(1), final_weight, 1e-5)

        for fobj in fobjs:
            fobj.close()

    def test_NuE(self):
        self.cmp_dataset("upgrade_genie_step3_140021_000000")


if __name__ == "__main__":
    unittest.main()
