#!/usr/bin/env python

import json
import os.path
import unittest

import numpy as np

from simweights import fluxes

E = np.logspace(2, 10, 9)


class TestCosmicRayModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.dirname(__file__) + "/flux_values.json", "r") as f:
            cls.flux_values = json.load(f)

    def flux_cmp(self, name):
        if name == "FixedFractionFlux":
            args = ({2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4},)
        else:
            args = ()
        flux = getattr(fluxes, name)(*args)
        v1 = flux(*np.meshgrid(E, [int(i) for i in self.flux_values[name].keys()]))
        v2 = np.array(list(self.flux_values[name].values()))

        m = v2 != 0
        np.testing.assert_array_almost_equal(v1, v2, 17)
        np.testing.assert_array_almost_equal(v1[m] / v2[m], 1, 13)

    def test_Hoerandel(self):
        self.flux_cmp("Hoerandel")

    def test_Hoerandel5(self):
        self.flux_cmp("Hoerandel5")

    def test_Hoerandel_IT(self):
        self.flux_cmp("Hoerandel_IT")

    def test_GaisserHillas(self):
        self.flux_cmp("GaisserHillas")

    def test_GaisserH3a(self):
        self.flux_cmp("GaisserH3a")

    def test_GaisserH4a(self):
        self.flux_cmp("GaisserH4a")

    def test_GaisserH4a_IT(self):
        self.flux_cmp("GaisserH4a_IT")

    def test_Honda2004(self):
        self.flux_cmp("Honda2004")

    def test_TIG1996(self):
        self.flux_cmp("TIG1996")

    def test_GlobalFitGST(self):
        self.flux_cmp("GlobalFitGST")

    def test_FixedFractionFlux(self):
        self.flux_cmp("FixedFractionFlux")

    def test_corsika_to_pdg(self):
        c = [
            14,
            402,
            703,
            904,
            1105,
            1206,
            1407,
            1608,
            1909,
            2010,
            2311,
            2412,
            2713,
            2814,
            3115,
            3216,
            3517,
            4018,
            3919,
            4020,
            4521,
            4822,
            5123,
            5224,
            5525,
            5626,
        ]
        pdgid = [int(i) for i in self.flux_values["Hoerandel"].keys()]
        np.testing.assert_array_equal(fluxes.corsika_to_pdg(c), pdgid)


if __name__ == "__main__":
    unittest.main()
