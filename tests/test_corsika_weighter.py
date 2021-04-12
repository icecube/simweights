#!/usr/bin/env python
import os
import unittest

import h5py
import numpy as np
import pandas
import tables
from scipy.integrate import quad
from scipy.interpolate import interp1d

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


def get_cos_zenith_dist(c, N):
    cz = np.linspace(c.cos_zen_min, c.cos_zen_max, 1000)
    cdf = (c._diff_etendue(cz) - c._diff_etendue(c.cos_zen_min)) / (
        c._diff_etendue(c.cos_zen_max) - c._diff_etendue(c.cos_zen_min)
    )
    cf = interp1d(cdf, cz)
    p = np.linspace(0, 1, N)
    return cf(p)


def make_corsika_data(pdgid, nevents, c, emin, emax, egamma):
    weight = np.zeros(nevents, dtype=weight_dtype)
    weight["ParticleType"] = pdgid
    weight["CylinderLength"] = c.length
    weight["CylinderRadius"] = c.radius
    weight["ThetaMin"] = np.arccos(c.cos_zen_max)
    weight["ThetaMax"] = np.arccos(c.cos_zen_min)
    weight["OverSampling"] = 1
    weight["Weight"] = 1
    weight["NEvents"] = nevents
    weight["EnergyPrimaryMin"] = emin
    weight["EnergyPrimaryMax"] = emax
    weight["PrimarySpectralIndex"] = egamma

    primary = np.zeros(nevents, dtype=primary_dtype)
    primary["type"] = pdgid
    primary["zenith"] = np.arccos(get_cos_zenith_dist(c, nevents))
    np.random.shuffle(primary["zenith"])

    if egamma == -1:
        primary["energy"] = np.geomspace(emin, emax, nevents)
    else:
        q = np.linspace(1 / 2 / nevents, 1 - 1 / 2 / nevents, nevents)
        G = egamma + 1
        primary["energy"] = (q * (emax ** G - emin ** G) + emin ** G) ** (1 / G)
    return dict(CorsikaWeightMap=weight, PolyplopiaPrimary=primary)


def make_hdf5_file(fname, v):
    d = make_corsika_data(*v)
    f = h5py.File(fname, "w")
    f.create_dataset("CorsikaWeightMap", data=d["CorsikaWeightMap"])
    f.create_dataset("PolyplopiaPrimary", data=d["PolyplopiaPrimary"])
    f.close()


class TestCorsikaWeighter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = simweights.NaturalRateCylinder(1200, 600, 0, 1)
        make_hdf5_file("file1.h5", (2212, 100000, cls.c, 1e4, 1e6, -1))
        make_hdf5_file("file2.h5", (2212, 100000, cls.c, 1e5, 1e7, -1.5))
        cls.flux_model1 = lambda cls, energy, pdgid: 1
        cls.flux_model2 = simweights.TIG1996()

    @classmethod
    def tearDownClass(cls):
        os.unlink("file1.h5")
        os.unlink("file2.h5")

    def check_weights(self, wf):
        w1 = wf.get_weights(self.flux_model1)
        emin, emax = wf.surface.get_energy_range(2212)
        self.assertAlmostEqual(w1.sum() / (emax - emin) / self.c.etendue, 1, 4)
        E = wf.get_column("PolyplopiaPrimary", "energy")
        y, x = np.histogram(E, weights=w1, bins=50, range=[emin, emax])
        Ewidth = x[1:] - x[:-1]
        np.testing.assert_array_almost_equal(y / Ewidth / self.c.etendue, 1, 2)

        w2 = wf.get_weights(self.flux_model2)
        surface2 = quad(self.flux_model2, emin, emax, args=(2212,))[0] * self.c.etendue
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

    def test_addition_energy(self):
        simfile1 = h5py.File("file1.h5", "r")
        simfile2 = pandas.HDFStore("file2.h5", "r")
        wf1 = simweights.CorsikaWeighter(simfile1, nfiles=1)
        wf2 = simweights.CorsikaWeighter(simfile2, nfiles=1)
        wf = wf1 + wf2
        self.check_weights(wf)

        self.assertEqual(type(wf1.surface), simweights.GenerationSurface)
        self.assertEqual(type(wf2.surface), simweights.GenerationSurface)
        self.assertEqual(type(wf.surface), simweights.GenerationSurfaceCollection)
        self.assertEqual(len(wf.surface.spectra[2212]), 2)

    def test_zenith(self):
        c1 = simweights.NaturalRateCylinder(1200, 600, 0, 0.7)
        c2 = simweights.NaturalRateCylinder(1200, 600, 0.3, 1)
        cc = simweights.NaturalRateCylinder(1200, 600, 0, 1)

        N = 100_000
        wf1 = simweights.CorsikaWeighter(make_corsika_data(2212, N, c1, 1e4, 1e6, 0), nfiles=1)
        wf2 = simweights.CorsikaWeighter(make_corsika_data(2212, N, c2, 1e4, 1e6, 0), nfiles=1)
        wfc = wf1 + wf2

        for wobj, c, n in [(wf1, c1, N), (wf2, c2, N), (wfc, cc, 2 * N)]:
            weights = wobj.get_weights(self.flux_model1)
            zen = wobj.get_column("PolyplopiaPrimary", "zenith")
            self.assertEqual(weights.size, n)
            self.assertEqual(zen.size, n)
            self.assertAlmostEqual(weights.sum() / (1e6 - 1e4) / c.etendue, 1, 5)

            y, x = np.histogram(wobj.get_column("PolyplopiaPrimary", "zenith"), weights=weights)
            for i, _ in enumerate(y):
                Integral = (
                    2
                    * np.pi
                    * (1e6 - 1e4)
                    * quad(lambda x: np.sin(x) * c1.projected_area(np.cos(x)), x[i], x[i + 1])[0]
                )
                self.assertAlmostEqual(y[i] / Integral, 1, 2)

        self.assertEqual(type(wf1.surface), simweights.GenerationSurface)
        self.assertEqual(type(wf2.surface), simweights.GenerationSurface)
        self.assertEqual(type(wfc.surface), simweights.GenerationSurfaceCollection)
        self.assertEqual(len(wfc.surface.spectra[2212]), 2)

    def test_effective_area_simple(self):
        for x in np.linspace(-1, 1, 21):
            c = simweights.NaturalRateCylinder(1000, 500, *sorted((x, x + (1 if x < 0 else -1) * 1e-9)))
            wf1 = simweights.CorsikaWeighter(make_corsika_data(2212, 1000, c, 1e3, 1e4, 0), nfiles=1)
            if abs(x) == 0:
                self.assertEqual(1000 * 500 * 2, c.projected_area(x))
            if abs(x) == 1:
                self.assertEqual(np.pi * 500 ** 2, c.projected_area(x))
            ea = wf1.effective_area(2212)
            self.assertEqual(ea.shape, (1, 1))
            self.assertAlmostEqual(ea[0, 0] / c.projected_area(x), 1, 4)

    def test_effective_area_binned(self):
        c = simweights.NaturalRateCylinder(1000, 500, -1, 1)
        wf1 = simweights.CorsikaWeighter(make_corsika_data(2212, 100000, c, 1e3, 1e4, 0), nfiles=1)

        ea = wf1.effective_area(2212)
        self.assertEqual(ea.shape, (1, 1))
        self.assertAlmostEqual(ea[0, 0] / c.etendue * 4 * np.pi, 1)

        eb = np.linspace(1e3, 1e4, 5)
        ea = wf1.effective_area(2212, energy_bins=eb)
        self.assertEqual(ea.shape, (1, 4))
        np.testing.assert_array_almost_equal(ea / c.etendue * 4 * np.pi, 1)

        czb = np.linspace(-1, 1, 21)
        detendue = np.array([c._diff_etendue(x) for x in czb])
        actual_etendue = np.ediff1d(detendue) / np.ediff1d(czb) / 2 / np.pi

        ea = wf1.effective_area(2212, cos_zenith_bins=czb)
        self.assertEqual(ea.shape, (20, 1))
        np.testing.assert_array_almost_equal(ea[:, 0] / actual_etendue, 1, 3)

        ea = wf1.effective_area(2212, energy_bins=eb, cos_zenith_bins=czb)
        aee = np.repeat(actual_etendue, 4).reshape(20, 4)
        self.assertEqual(ea.shape, (20, 4))
        np.testing.assert_array_almost_equal(ea / aee, 1, 1)

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
