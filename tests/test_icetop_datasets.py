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

from simweights import IceTopWeighter

datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
datasets = ["Level3_IC86.2012_SIBYLL2.1_p_12360_E6.0_0"]
approx = pytest.approx

loaders = [
    pytest.param(lambda f: h5py.File(str(f) + ".hdf5", "r"), id="h5py"),
    pytest.param(lambda f: uproot.open(str(f) + ".root"), id="uproot"),
    pytest.param(lambda f: tables.open_file(str(f) + ".hdf5", "r"), id="pytables"),
    pytest.param(lambda f: pd.HDFStore(str(f) + ".hdf5", "r"), id="pandas"),
]


def load_reference_values(filename):
    reffile = h5py.File(str(filename) + ".hdf5", "r")
    ref = {}

    assert len(reffile["I3TopInjectorInfo"]) == 1
    si = reffile["I3TopInjectorInfo"][0]
    pri = reffile["MCPrimary"]
    ref["solid_angle"] = np.pi * (np.cos(si["min_zenith"]) ** 2 - np.cos(si["max_zenith"]) ** 2)
    ref["injection_area"] = np.pi * (si["sampling_radius"] * 1e2) ** 2
    ref["energy_integral"] = np.log(si["max_energy"] / si["min_energy"])  # assuming E^-1
    ref["energy_factor"] = ref["energy_integral"] * pri["energy"]
    ref["final_weight"] = ref["energy_factor"] * ref["solid_angle"] * ref["injection_area"] / si["n_events"]
    return ref


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.parametrize("loader", loaders)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(fname, loader):
    filename = Path(datadir) / fname

    ref = load_reference_values(filename)
    fobj = loader(filename)
    w = IceTopWeighter(fobj)
    spatial = w.surface.spectra[2212][0].dists[1]
    proj_area = spatial.projected_area(1)
    assert proj_area == approx(ref["injection_area"])
    sw_etendue = 1 / spatial.pdf(1)
    assert sw_etendue == approx(ref["solid_angle"] * ref["injection_area"], 1e-5)
    assert ref["energy_factor"] == approx(ref["energy_integral"] * w.get_weight_column("energy"))
    assert ref["final_weight"] == approx(w.get_weights(1), 1e-5)

    fobj.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
