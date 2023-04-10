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
from simweights import CorsikaWeighter, GaisserH4a
from simweights._utils import constcol


class TestCorsikaDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux = GaisserH4a()
        datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
        if not datadir:
            cls.skipTest(None, "environment variable SIMWEIGHTS_TESTDATA not set")
        cls.datadir = datadir + "/"

    def untriggered_weights(self, f):
        cwm = f["CorsikaWeightMap"]
        pflux = self.flux(cwm["PrimaryEnergy"], cwm["PrimaryType"])
        if (cwm["PrimarySpectralIndex"] == -1).any():
            assert (cwm["PrimarySpectralIndex"] == -1).all()
            energy_integral = np.log(cwm["EnergyPrimaryMax"] / cwm["EnergyPrimaryMin"])
        else:
            energy_integral = (
                cwm["EnergyPrimaryMax"] ** (cwm["PrimarySpectralIndex"] + 1)
                - cwm["EnergyPrimaryMin"] ** (cwm["PrimarySpectralIndex"] + 1)
            ) / (cwm["PrimarySpectralIndex"] + 1)

        energy_weight = cwm["PrimaryEnergy"] ** cwm["PrimarySpectralIndex"]
        return (
            1e4
            * pflux
            * energy_integral
            / energy_weight
            * cwm["AreaSum"]
            / (cwm["NEvents"] * cwm["OverSampling"])
        )

    def triggered_weights(self, f):
        i3cw = f["I3CorsikaWeight"]
        flux_val = self.flux(i3cw["energy"], i3cw["type"])
        info = f["I3PrimaryInjectorInfo"]
        energy = i3cw["energy"]
        epdf = np.zeros_like(energy, dtype=float)

        for ptype in np.unique(info["primary_type"]):
            info_mask = info["primary_type"] == ptype
            n_events = info["n_events"][info_mask].sum()
            min_energy = constcol(info, "min_energy", info_mask)
            max_energy = constcol(info, "max_energy", info_mask)
            min_zenith = constcol(info, "min_zenith", info_mask)
            max_zenith = constcol(info, "max_zenith", info_mask)
            cylinder_height = constcol(info, "cylinder_height", info_mask)
            cylinder_radius = constcol(info, "cylinder_radius", info_mask)
            power_law_index = constcol(info, "power_law_index", info_mask)

            G = power_law_index + 1
            side = 2e4 * cylinder_radius * cylinder_height
            cap = 1e4 * np.pi * cylinder_radius**2
            cos_minz = np.cos(min_zenith)
            cos_maxz = np.cos(max_zenith)
            ET1 = cap * cos_minz * np.abs(cos_minz) + side * (
                cos_minz * np.sqrt(1 - cos_minz**2) - min_zenith
            )
            ET2 = cap * cos_maxz * np.abs(cos_maxz) + side * (
                cos_maxz * np.sqrt(1 - cos_maxz**2) - max_zenith
            )
            etendue = np.pi * (ET1 - ET2)

            mask = ptype == i3cw["type"]
            energy_term = energy[mask] ** power_law_index * G / (max_energy**G - min_energy**G)
            epdf[mask] += n_events * energy_term / etendue

        return i3cw["weight"] * flux_val / epdf

    def cmp_dataset(self, triggered, fname, rate):
        fname = self.datadir + fname

        if triggered:
            nfiles = None
            refweight = self.triggered_weights
        else:
            nfiles = 1
            refweight = self.untriggered_weights

        reffile = h5py.File(fname + ".hdf5", "r")
        w0 = refweight(reffile)

        inputfiles = [
            ("h5py", reffile),
            ("uproot", uproot.open(fname + ".root")),
            ("tables", tables.open_file(fname + ".hdf5", "r")),
            ("pandas", pd.HDFStore(fname + ".hdf5", "r")),
        ]

        for name, infile in inputfiles:
            with self.subTest(name):
                wobj = CorsikaWeighter(infile, nfiles)
                w = wobj.get_weights(self.flux)
                self.assertAlmostEqual(w.sum(), rate)
                np.testing.assert_allclose(w0, w, 1e-6)

        for _, infile in inputfiles:
            infile.close()

    def test_012602(self):
        self.cmp_dataset(False, "Level2_IC86.2015_corsika.012602.000000", 102.01712611701736)

    def test_020014(self):
        self.cmp_dataset(False, "Level2_IC86.2015_corsika.020014.000000", 23.015500214424705)

    def test_020021(self):
        self.cmp_dataset(False, "Level2_IC86.2015_corsika.020021.000000", 69.75465614509928)

    def test_020208(self):
        self.cmp_dataset(False, "Level2_IC86.2016_corsika.020208.000001", 22.622983704306385)

    def test_020243(self):
        self.cmp_dataset(False, "Level2_IC86.2016_corsika.020243.000001", 4.590586137762489)

    def test_020263(self):
        self.cmp_dataset(False, "Level2_IC86.2016_corsika.020263.000000", 10.183937153798436)

    def test_020777(self):
        self.cmp_dataset(False, "Level2_IC86.2016_corsika.020777.000000", 362.94284441826704)

    def test_020778(self):
        self.cmp_dataset(False, "Level2_IC86.2016_corsika.020778.000000", 6.2654796956603)

    def test_020780(self):
        self.cmp_dataset(False, "Level2_IC86.2016_corsika.020780.000000", 14.215947086098588)

    def test_21889(self):
        self.cmp_dataset(True, "Level2_IC86.2016_corsika.021889.000000", 122.83809329321922)


if __name__ == "__main__":
    unittest.main()
