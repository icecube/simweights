#!/usr/bin/env python
import os
import unittest

import numpy as np
import pandas as pd

from simweights import GenieWeighter


class TestNugenDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        datadir = os.environ.get("SIMWEIGHTS_DATA", None)
        if not datadir:
            cls.skipTest(None, "environment variable SIMWEIGHTS_DATA not set")
        cls.datadir = datadir + "/"

    def cmp_dataset(self, fname):
        f = pd.HDFStore(self.datadir + "/" + fname, "r")
        wd = f["I3MCWeightDict"]
        w = GenieWeighter(f)

        solid_angle = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))

        cylinder = w.surface.spatial_dist
        proj_area = cylinder.projected_area(0)
        injection_area = np.pi * (wd["InjectionSurfaceR"] * 1e2) ** 2
        np.testing.assert_allclose(proj_area, injection_area)

        sw_etendue = 1 / cylinder.pdf(0)
        np.testing.assert_allclose(sw_etendue, solid_angle * injection_area, 1e-5)

        power_law = w.surface.energy_dist
        energy_factor = 1 / power_law.pdf(wd["PrimaryNeutrinoEnergy"])
        total_weight = wd["TotalInteractionProbabilityWeight"]

        one_weight = total_weight * energy_factor * solid_angle * injection_area
        np.testing.assert_allclose(one_weight, wd["OneWeight"])

        one_weight = total_weight / power_law.pdf(wd["PrimaryNeutrinoEnergy"]) / cylinder.pdf(0)
        np.testing.assert_allclose(one_weight, wd["OneWeight"], 1e-5)

        np.testing.assert_allclose(
            w.get_weights(1), wd["OneWeight"] / (f["I3GenieInfo"]["n_flux_events"][0]), 1e-5
        )

        f.close()

    def test_NuE(self):
        self.cmp_dataset("genie_reader_NuE_C_corr.hdf5")


if __name__ == "__main__":
    unittest.main()
