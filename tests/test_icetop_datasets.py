#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import os
import unittest

import h5py
import numpy as np
import pandas as pd
import tables
import uproot

from simweights import IceTopWeighter, PowerLaw
from simweights._utils import get_column, get_table


class TestIceTopDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
        if not datadir:
            cls.skipTest(None, "environment variable SIMWEIGHTS_TESTDATA not set")
        cls.datadir = datadir + "/"

    def cmp_dataset(self, fname):
        filename = os.path.join(self.datadir, fname)
        reffile = pd.HDFStore(filename + ".hdf5", "r")

        assert len(reffile["I3TopInjectorInfo"]) == 1
        si = reffile["I3TopInjectorInfo"].iloc[0]
        pri = reffile["MCPrimary"]
        print(si)
        solid_angle = 2 * np.pi * (np.cos(si["min_zenith"]) - np.cos(si["max_zenith"]))
        injection_area = np.pi * (si["sampling_radius"] * 1e2) ** 2
        pl = PowerLaw(si["power_law_index"], si["min_energy"], si["max_energy"])
        energy_factor = 1 / pl.pdf(pri["energy"])
        final_weight = energy_factor * solid_angle * injection_area / si["n_events"]

        fobjs = [
            h5py.File(filename + ".hdf5", "r"),
            tables.open_file(filename + ".hdf5", "r"),
            reffile,
            uproot.open(filename + ".root"),
        ]

        for fobj in fobjs:
            with self.subTest(lib=str(fobj)):

                w = IceTopWeighter(fobj)

                spatial = w.surface.spectra[2212][0].spatial_dist
                proj_area = spatial.projected_area(1)
                np.testing.assert_allclose(proj_area, injection_area)

                sw_etendue = 1 / spatial.pdf(1)
                np.testing.assert_allclose(sw_etendue, solid_angle * injection_area, 1e-5)

                power_law = w.surface.spectra[2212][0].energy_dist
                energy_term = 1 / power_law.pdf(w.get_weight_column("energy"))
                np.testing.assert_allclose(energy_term, energy_factor)

                np.testing.assert_allclose(w.get_weights(1), final_weight, 1e-5)

        for fobj in fobjs:
            fobj.close()

    def test_NuE(self):
        self.cmp_dataset("icetop_testoutput")


if __name__ == "__main__":
    unittest.main()