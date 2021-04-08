#!/usr/bin/env python
import os
import unittest

import h5py
import numpy as np
import pandas
import tables
from scipy.integrate import quad

import simweights

primary_dtype = [("type", np.int32), ("energy", np.float64), ("zenith", np.float64)]

weight_dtype = [
    ("ParticleType", np.int32),
    ("CylinderLength", np.float64),
    ("CylinderRadius", np.float64),
    ("ThetaMin", np.float64),
    ("ThetaMax", np.float64),
    ("OverSampling", np.float64),
    ("Weight", np.float64),
    ("NEvents", np.float64),
    ("EnergyPrimaryMin", np.float64),
    ("EnergyPrimaryMax", np.float64),
    ("PrimarySpectralIndex", np.float64),
]


def make_hdf5_file(fname, v):
    weight = np.zeros(v[1], dtype=weight_dtype)
    weight["ParticleType"] = v[0]
    weight["CylinderLength"] = v[2]
    weight["CylinderRadius"] = v[3]
    weight["ThetaMin"] = v[4]
    weight["ThetaMax"] = v[5]
    weight["OverSampling"] = 50
    weight["Weight"] = 1
    weight["NEvents"] = v[1] / 50.0
    weight["EnergyPrimaryMin"] = v[6]
    weight["EnergyPrimaryMax"] = v[7]
    weight["PrimarySpectralIndex"] = v[8]

    primary = np.zeros(v[1], dtype=primary_dtype)
    primary["type"] = v[0]
    primary["zenith"] = np.linspace(v[4], v[5], v[1])
    if v[8] == -1:
        primary["energy"] = np.geomspace(v[6], v[7], v[1])
    else:
        q = np.linspace(1 / 2 / v[1], 1 - 1 / 2 / v[1], v[1])
        G = v[8] + 1
        primary["energy"] = (q * (v[7] ** G - v[6] ** G) + v[6] ** G) ** (1 / G)

    f = h5py.File(fname, "w")
    f.create_dataset("CorsikaWeightMap", data=weight)
    f.create_dataset("PolyplopiaPrimary", data=primary)
    f.close()


class TestCorsikaWeighter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_hdf5_file("file1.h5", (2212, 100000, 1200, 600, 0, np.pi, 1e4, 1e6, -1))
        make_hdf5_file("file2.h5", (2212, 100000, 1200, 600, 0, np.pi, 1e5, 1e7, -1.5))
        cls.etendue = simweights.NaturalRateCylinder(600, 1200, 0, 1).etendue
        cls.flux_model1 = lambda cls, energy, pdgid: 1 / cls.etendue
        cls.flux_model2 = simweights.TIG1996()

    @classmethod
    def tearDownClass(cls):
        os.unlink("file1.h5")
        os.unlink("file2.h5")

    def check_weights(self, wf):
        w1 = wf.get_weights(self.flux_model1)
        emin, emax = wf.surface.get_energy_range(2212)
        self.assertAlmostEqual(w1.sum() / (emax - emin), 1, 4)
        E = wf.get_column("PolyplopiaPrimary", "energy")
        y, x = np.histogram(E, weights=w1, bins=50, range=[emin, emax])
        Ewidth = x[1:] - x[:-1]
        np.testing.assert_array_almost_equal(y / Ewidth, 1, 2)

        w2 = wf.get_weights(self.flux_model2)
        surface2 = quad(self.flux_model2, emin, emax, args=(2212,))[0] * self.etendue
        self.assertAlmostEqual(w2.sum() / surface2, 1, 4)

    def test_h5py(self):
        simfile = h5py.File("file1.h5", "r")
        wf = simweights.CorsikaWeighter(simfile, nfiles=1)
        self.check_weights(wf)

    def test_pytables(self):
        simfile = tables.open_file("file2.h5", "r")
        wf = simweights.CorsikaWeighter(simfile, nfiles=1)
        self.check_weights(wf)

    def test_pandas(self):
        simfile = pandas.HDFStore("file1.h5", "r")
        wf = simweights.CorsikaWeighter(simfile, nfiles=1)
        self.check_weights(wf)

    def test_addition(self):
        simfile1 = h5py.File("file1.h5", "r")
        simfile2 = pandas.HDFStore("file2.h5", "r")
        wf1 = simweights.CorsikaWeighter(simfile1, nfiles=1)
        wf2 = simweights.CorsikaWeighter(simfile2, nfiles=1)
        wf = wf1 + wf2
        self.check_weights(wf)

    def test_outside(self):
        # make sure we give a warning if energy or zenith angle is out of bounds
        wei = np.array([(2212, 1200, 600, 0, np.pi / 2, 1, 1, 1, 1e3, 1e4, -2)], dtype=weight_dtype)
        pri = np.array([(2212, 3e2, 1)], dtype=primary_dtype)
        x = dict(CorsikaWeightMap=wei, PolyplopiaPrimary=pri)
        wf1 = simweights.CorsikaWeighter(x, nfiles=1)
        with self.assertWarns(UserWarning):
            wf1.get_weights(self.flux_model1)

        wei = np.array([(2212, 1200, 600, 0, np.pi / 2, 1, 1, 1, 1e3, 1e4, -2)], dtype=weight_dtype)
        pri = np.array([(2212, 3e3, 1.6)], dtype=primary_dtype)
        x = dict(CorsikaWeightMap=wei, PolyplopiaPrimary=pri)
        wf1 = simweights.CorsikaWeighter(x, nfiles=1)
        with self.assertWarns(UserWarning):
            wf1.get_weights(self.flux_model1)


if __name__ == "__main__":
    unittest.main()
