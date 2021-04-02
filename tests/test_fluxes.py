#!/usr/bin/env python

import json
import unittest

import numpy as np

from simweights import fluxes

E = np.logspace(2, 10, 9)


class TestCosmicRayModels(unittest.TestCase):
    def flux_cmp(self, name):

        #        f = open('flux_values.json','r')
        # flux_values = json.load(f)

        if name == "FixedFractionFlux":
            args = ({2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4},)
        else:
            args = ()
        flux = getattr(fluxes, name)(*args)

        print(E)
        print(list(flux_values[name].keys()))
        v1 = flux(*np.meshgrid(E, [int(i) for i in flux_values[name].keys()]))
        v2 = np.array(list(flux_values[name].values()))

        print(v1)
        print(v2)

        assert (abs(v1 - v2) < 1e-17).all()
        assert (abs(v1[v2 != 0] / v2[v2 != 0] - 1) < 1e-13).all()

    def test_corsika_to_pdg(self):
        c = [   14,  402,  703,  904, 1105, 1206, 1407, 1608, 1909, 2010, 2311, 2412, 2713,
              2814, 3115, 3216, 3517, 4018, 3919, 4020, 4521, 4822, 5123, 5224, 5525, 5626 ]
        pdg = [int(i) for i in flux_values["Hoerandel"].keys()]
        assert np.all(fluxes.corsika_to_pdg(c) == pdg)


with open("flux_values.json", "r") as f:
    flux_values = json.load(f)

for m in flux_values.keys():
    setattr(TestCosmicRayModels, "test_" + m, lambda self, m=m: self.flux_cmp(m))

if __name__ == "__main__":
    unittest.main()
