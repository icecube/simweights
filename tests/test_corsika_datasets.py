#!/usr/bin/env python
import os
import unittest

import numpy as np
import tables

from simweights import CorsikaWeighter, GaisserH4a


class TestCorsikaDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux = GaisserH4a()
        datadir = os.environ.get("SIMWEIGHTS_DATA", None)
        if not datadir:
            cls.skipTest(None, "enviornment varible SIMWEIGHTS_DATA not set")
        cls.datadir = datadir + "/"

    def cmp_dataset(self, fname, rate):
        fname = self.datadir + fname
        f = tables.open_file(fname, "r")
        wo = CorsikaWeighter(f, nfiles=1)
        w1 = wo.get_weights(self.flux)

        self.assertAlmostEqual(w1.sum(), rate)

        cwm = f.root.CorsikaWeightMap
        pflux = self.flux(cwm.cols.PrimaryEnergy[:], cwm.cols.PrimaryType[:])
        if (cwm.cols.PrimarySpectralIndex[:] == -1).any():
            assert (cwm.cols.PrimarySpectralIndex[:] == -1).all()
            energy_integral = np.log(cwm.cols.EnergyPrimaryMax[:] / cwm.cols.EnergyPrimaryMin[:])
        else:
            energy_integral = (
                cwm.cols.EnergyPrimaryMax[:] ** (cwm.cols.PrimarySpectralIndex[:] + 1)
                - cwm.cols.EnergyPrimaryMin[:] ** (cwm.cols.PrimarySpectralIndex[:] + 1)
            ) / (cwm.cols.PrimarySpectralIndex[:] + 1)

        energy_weight = cwm.cols.PrimaryEnergy[:] ** cwm.cols.PrimarySpectralIndex[:]
        w2 = (
            1e4
            * pflux
            * energy_integral
            / energy_weight
            * cwm.cols.AreaSum[:]
            / (cwm.cols.NEvents[:] * cwm.cols.OverSampling[:])
        )
        np.testing.assert_allclose(w1, w2, 1e-6)

        f.close()

    def test_20904(self):
        # low-level-ml minbias
        self.cmp_dataset("CORSIKA_20904_minbias_1.hdf5", 4419.067527326577)

    def test_020777(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020777.000000.hdf5", 362.94284441826704)

    def test_020780(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020780.000000.hdf5", 14.215947086098588)

    def test_020778(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020778.000000.hdf5", 6.2654796956603)

    def test_020263(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020263.000000.hdf5", 10.183937153798436)

    def test_020243(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020243.000001.hdf5", 4.590586137762489)

    def test_020040(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020040.084027.hdf5", 1.2340187693531177)

    def test_020208(self):
        self.cmp_dataset("Level2_IC86.2016_corsika.020208.000001.hdf5", 22.622983704306385)

    def test_020021(self):
        self.cmp_dataset("Level2_IC86.2015_corsika.020021.000000.hdf5", 69.75465614509928)

    def test_012602(self):
        self.cmp_dataset("Level2_IC86.2015_corsika.012602.000000.hdf5", 102.01712611701736)

    def test_020014(self):
        self.cmp_dataset("Level2_IC86.2015_corsika.020014.000000.hdf5", 23.015500214424705)


if __name__ == "__main__":
    unittest.main()
