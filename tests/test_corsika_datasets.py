#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import tables
import uproot

from simweights import CorsikaWeighter, GaisserH4a
from simweights._utils import constcol

flux = GaisserH4a()
datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
if datadir:
    datadir = Path(datadir)

datasets = [
    pytest.param(False, "Level2_IC86.2015_corsika.012602.000000", 101.71983520393832, id="12602"),
    pytest.param(False, "Level2_IC86.2015_corsika.020014.000000", 22.857025444108743, id="20014"),
    pytest.param(False, "Level2_IC86.2015_corsika.020021.000000", 68.93732262681625, id="20021"),
    pytest.param(False, "Level2_IC86.2016_corsika.020208.000001", 12.397742530207822, id="20208"),
    pytest.param(False, "Level2_IC86.2016_corsika.020243.000001", 3.302275062730073, id="20243"),
    pytest.param(False, "Level2_IC86.2016_corsika.020263.000000", 5.3137132171197905, id="20263"),
    pytest.param(False, "Level2_IC86.2016_corsika.020777.000000", 359.20422121174204, id="20777"),
    pytest.param(False, "Level2_IC86.2016_corsika.020778.000000", 6.25969855736358, id="20778"),
    pytest.param(False, "Level2_IC86.2016_corsika.020780.000000", 13.864780296171585, id="20780"),
    pytest.param(True, "Level2_IC86.2016_corsika.021889.000000", 122.56278334422919, id="21889"),
]

loaders = [
    pytest.param(lambda f: h5py.File(str(f) + ".hdf5", "r"), id="h5py"),
    pytest.param(lambda f: uproot.open(str(f) + ".root"), id="uproot"),
    pytest.param(lambda f: tables.open_file(str(f) + ".hdf5", "r"), id="pytables"),
    pytest.param(
        lambda f: pd.HDFStore(str(f) + ".hdf5", "r"),
        id="pandas",
    ),
]


def untriggered_weights(f):
    cwm = f["CorsikaWeightMap"]
    pflux = flux(cwm["PrimaryEnergy"], cwm["PrimaryType"])
    if (cwm["PrimarySpectralIndex"] == -1).any():
        assert (cwm["PrimarySpectralIndex"] == -1).all()
        energy_integral = np.log(cwm["EnergyPrimaryMax"] / cwm["EnergyPrimaryMin"])
    else:
        energy_integral = (
            cwm["EnergyPrimaryMax"] ** (cwm["PrimarySpectralIndex"] + 1)
            - cwm["EnergyPrimaryMin"] ** (cwm["PrimarySpectralIndex"] + 1)
        ) / (cwm["PrimarySpectralIndex"] + 1)

    energy_weight = cwm["PrimaryEnergy"] ** cwm["PrimarySpectralIndex"]
    return 1e4 * pflux * energy_integral / energy_weight * cwm["AreaSum"] / (cwm["NEvents"] * cwm["OverSampling"])


def triggered_weights(f):
    i3cw = f["I3CorsikaWeight"]
    flux_val = flux(i3cw["energy"], i3cw["type"])
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
        ET1 = cap * cos_minz * np.abs(cos_minz) + side * (cos_minz * np.sqrt(1 - cos_minz**2) - min_zenith)
        ET2 = cap * cos_maxz * np.abs(cos_maxz) + side * (cos_maxz * np.sqrt(1 - cos_maxz**2) - max_zenith)
        etendue = np.pi * (ET1 - ET2)

        mask = ptype == i3cw["type"]
        energy_term = energy[mask] ** power_law_index * G / (max_energy**G - min_energy**G)
        epdf[mask] += n_events * energy_term / etendue

    return i3cw["weight"] * flux_val / epdf


@pytest.mark.parametrize(("triggered", "fname", "rate"), datasets)
@pytest.mark.parametrize("loader", loaders)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(triggered, fname, rate, loader):
    fname = datadir / fname

    if triggered:
        nfiles = None
        refweight = triggered_weights
    else:
        nfiles = 1
        refweight = untriggered_weights

    reffile = h5py.File(str(fname) + ".hdf5", "r")
    w0 = refweight(reffile)

    infile = loader(fname)
    wobj = CorsikaWeighter(infile, nfiles)
    w = wobj.get_weights(flux)
    assert w.sum() == pytest.approx(rate)
    assert w0 == pytest.approx(w, 1e-6)
    infile.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
