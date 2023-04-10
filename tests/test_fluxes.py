#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import json
import unittest
from pathlib import Path

import numpy as np
from simweights import _fluxes

E = np.logspace(2, 10, 9)


class TestCosmicRayModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (Path(__file__).parent / "flux_values.json").open() as f:
            cls.flux_values = json.load(f)

    def flux_cmp(self, name, *args):
        flux = getattr(_fluxes, name)(*args)
        v1 = flux(*np.meshgrid(E, [int(i) for i in self.flux_values[name]]))
        v2 = np.array(list(self.flux_values[name].values())) / 1e4

        np.testing.assert_allclose(v1, v2, 1e-13)

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
        self.flux_cmp("FixedFractionFlux", {2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4})
        self.flux_cmp(
            "FixedFractionFlux",
            {2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4},
            _fluxes.GaisserH4a_IT(),
        )


if __name__ == "__main__":
    unittest.main()
