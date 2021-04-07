#!/usr/bin/env python
import os
import unittest

import h5py
import numpy as np
import pandas
import tables

from simweights import NuGenWeighter, NaturalRateCylinder

weight_dtype = [
    ("PrimaryNeutrinoType", np.int32),
    ("NEvents", np.float64),
    ("CylinderHeight", np.float64),
    ("CylinderRadius", np.float64),
    ("MinZenith", np.float64),
    ("MaxZenith", np.float64),
    ("MinEnergyLog", np.float64),
    ("MaxEnergyLog", np.float64),
    ("PowerLawIndex", np.float64),
    ("PrimaryNeutrinoZenith", np.float64),
    ("PrimaryNeutrinoEnergy", np.float64),
    ("TotalWeight", np.float64),
]


def make_hdf5_file(fname, v):
    weight = np.zeros(v[1], dtype=weight_dtype)
    weight["NEvents"] = v[1]
    weight["PrimaryNeutrinoType"] = v[0]
    weight["CylinderHeight"] = v[2]
    weight["CylinderRadius"] = v[3]
    weight["MinZenith"] = v[4]
    weight["MaxZenith"] = v[5]
    weight["MinEnergyLog"] = np.log10(v[6])
    weight["MaxEnergyLog"] = np.log10(v[7])
    weight["PowerLawIndex"] = -v[8]
    weight["PrimaryNeutrinoZenith"] = np.linspace(v[4], v[5], v[1])
    weight["TotalWeight"] = 1
    if v[8] == -1:
        weight["PrimaryNeutrinoEnergy"] = np.geomspace(v[6], v[7], v[1])
    else:
        q = np.linspace(1 / 2 / v[1], 1 - 1 / 2 / v[1], v[1])
        G = v[8] + 1
        weight["PrimaryNeutrinoEnergy"] = (q * (v[7] ** G - v[6] ** G) + v[6] ** G) ** (1 / G)

    f = h5py.File(fname, "w")
    f.create_dataset("I3MCWeightDict", data=weight)
    f.close()


class TestCorsikaWeighter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_hdf5_file("file1.h5", (12, 100000, 1200, 600, 0, np.pi, 1e4, 1e6, -1))
        make_hdf5_file("file2.h5", (12, 100000, 1200, 600, 0, np.pi, 1e5, 1e7, -1.5))
        cls.etendue = NaturalRateCylinder(600, 1200, 0, 1).etendue
        cls.flux_model = lambda cls, energy, pdgid, cos_zen: 1 / cls.etendue

    @classmethod
    def tearDownClass(cls):
        os.unlink("file1.h5")
        os.unlink("file2.h5")

    def check_weights(self, wf):
        w = wf.get_weights(self.flux_model)
        emin, emax = wf.surface.get_energy_range(12)
        self.assertAlmostEqual(w.sum() / (emax - emin), 2, 4)
        E = wf.get_column("I3MCWeightDict", "PrimaryNeutrinoEnergy")
        y, x = np.histogram(E, weights=w, bins=50, range=[emin, emax])
        Ewidth = x[1:] - x[:-1]
        np.testing.assert_array_almost_equal(y / Ewidth, 2, 2)

    def test_h5py(self):
        simfile = h5py.File("file1.h5", "r")
        wf = NuGenWeighter(simfile, nfiles=1)
        self.check_weights(wf)

    def test_pytables(self):
        simfile = tables.open_file("file2.h5", "r")
        wf = NuGenWeighter(simfile, nfiles=1)
        self.check_weights(wf)

    def test_pandas(self):
        simfile = pandas.HDFStore("file1.h5", "r")
        wf = NuGenWeighter(simfile, nfiles=1)
        self.check_weights(wf)

    def test_addition(self):
        simfile1 = h5py.File("file1.h5", "r")
        simfile2 = pandas.HDFStore("file2.h5", "r")
        wf1 = NuGenWeighter(simfile1, nfiles=1)
        wf2 = NuGenWeighter(simfile2, nfiles=1)
        wf = wf1 + wf2
        self.check_weights(wf)


if __name__ == "__main__":
    unittest.main()
