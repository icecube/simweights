#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
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

with contextlib.suppress(ImportError):
    from icecube import dataio, simclasses  # noqa: F401

flux = GaisserH4a()
datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
if datadir:
    datadir = Path(datadir)

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


datasets = [
    pytest.param(untriggered_weights, 1, "Level2_IC86.2016_corsika.020208.000001", 12.397742530207822, id="20208"),
    pytest.param(untriggered_weights, 1, "Level2_IC86.2016_corsika.020243.000001", 3.302275062730073, id="20243"),
    pytest.param(untriggered_weights, 1, "Level2_IC86.2016_corsika.020263.000000", 5.3137132171197905, id="20263"),
    pytest.param(untriggered_weights, 1, "Level2_IC86.2016_corsika.020777.000000", 359.20422121174204, id="20777"),
    pytest.param(untriggered_weights, 1, "Level2_IC86.2016_corsika.020778.000000", 6.25969855736358, id="20778"),
    pytest.param(untriggered_weights, 1, "Level2_IC86.2016_corsika.020780.000000", 13.864780296171585, id="20780"),
    pytest.param(triggered_weights, None, "Level2_IC86.2016_corsika.021889.000000", 122.56278334422919, id="21889"),
    pytest.param(untriggered_weights, None, "Level2_IC86.2024_corsika.023111.000000", 2494.4260561524957, id="023111"),
]


@pytest.mark.parametrize(("refweight", "nfiles", "fname", "rate"), datasets)
@pytest.mark.parametrize("loader", loaders)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(refweight, nfiles, fname, rate, loader):
    fname = datadir / fname

    reffile = h5py.File(str(fname) + ".hdf5", "r")
    w0 = refweight(reffile)

    infile = loader(fname)
    wobj = CorsikaWeighter(infile, nfiles)
    w = wobj.get_weights(flux)
    assert w.sum() == pytest.approx(rate)
    assert w0 == pytest.approx(w, 1e-6)
    infile.close()


@pytest.mark.parametrize(("refweight", "nfiles", "fname", "rate"), datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
@pytest.mark.skipif("dataio" not in globals(), reason="Not in an IceTray environment")
def test_dataset_i3file(refweight, nfiles, fname, rate):
    fname = datadir / fname

    reffile = h5py.File(str(fname) + ".hdf5", "r")
    if "I3PrimaryInjectorInfo" in reffile:
        counts = np.unique(reffile["I3PrimaryInjectorInfo"]["primary_type"], return_counts=True)
        s_frame_counts = {counts[0][i]: counts[1][i] for i in range(len(counts[0]))}
    else:
        s_frame_counts = dict.fromkeys(set(reffile["CorsikaWeightMap"]["PrimaryType"]), 1)

    w0 = refweight(reffile)
    f = dataio.I3File(str(fname) + ".i3.zst")
    i = 0
    W = 0
    while f.more():
        frame = f.pop_frame()
        if frame.Stop != frame.DAQ:
            continue
        ww = CorsikaWeighter(frame, nfiles)
        pdgid = ww.get_weight_column("pdgid")[0]
        w = ww.get_weights(flux) / s_frame_counts[pdgid]
        assert w == pytest.approx(w0[i], 1e-6)
        i += 1
        W += w

    assert rate == pytest.approx(W, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
