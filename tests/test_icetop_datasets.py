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


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(fname):
    filename = Path(datadir) / fname
    reffile = h5py.File(str(filename) + ".hdf5", "r")

    assert len(reffile["I3TopInjectorInfo"]) == 1
    si = reffile["I3TopInjectorInfo"][0]
    pri = reffile["MCPrimary"]
    solid_angle = np.pi * (np.cos(si["min_zenith"]) ** 2 - np.cos(si["max_zenith"]) ** 2)
    injection_area = np.pi * (si["sampling_radius"] * 1e2) ** 2
    energy_integral = np.log(si["max_energy"] / si["min_energy"])  # assuming E^-1
    energy_factor = energy_integral * pri["energy"]
    final_weight = energy_factor * solid_angle * injection_area / si["n_events"]

    fobjs = [
        reffile,
        uproot.open(str(filename) + ".root"),
        tables.open_file(str(filename) + ".hdf5", "r"),
        pd.HDFStore(str(filename) + ".hdf5", "r"),
    ]

    for fobj in fobjs:
        w = IceTopWeighter(fobj)
        spatial = w.surface.spectra[2212][0].dists[1]
        proj_area = spatial.projected_area(1)
        assert proj_area == approx(injection_area)
        sw_etendue = 1 / spatial.pdf(1)
        assert sw_etendue == approx(solid_angle * injection_area, 1e-5)
        assert energy_factor == approx(energy_integral * w.get_weight_column("energy"))
        assert final_weight == approx(w.get_weights(1), 1e-5)

    for fobj in fobjs:
        fobj.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
