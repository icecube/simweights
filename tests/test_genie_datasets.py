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

from simweights import GenieWeighter
from simweights._utils import get_column, get_table

datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
datasets = [
    "upgrade_genie_step3_140021_000000",
    "upgrade_genie_step3_141828_000000",
    "GENIE_NuMu_IceCubeUpgrade_v58.22590.000000",
]
approx = pytest.approx


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(fname):
    filename = Path(datadir) / fname
    reffile = h5py.File(str(filename) + ".hdf5", "r")
    wd = reffile["I3MCWeightDict"]

    solid_angle = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))
    injection_area = np.pi * (wd["InjectionSurfaceR"] * 1e2) ** 2
    global_probability_scale = wd["GlobalProbabilityScale"]
    genie_weight = wd["GENIEWeight"]

    pli = -wd["PowerLawIndex"][0]
    energy_integral = ((10 ** wd["MaxEnergyLog"][0]) ** (pli + 1) - (10 ** wd["MinEnergyLog"][0]) ** (pli + 1)) / (pli + 1)
    energy_factor = 1 / (wd["PrimaryNeutrinoEnergy"] ** pli / energy_integral)
    one_weight = global_probability_scale * genie_weight * energy_factor * solid_angle * injection_area
    np.testing.assert_allclose(one_weight, wd["OneWeight"])
    final_weight = wd["OneWeight"] / (get_column(get_table(reffile, "I3GenieInfo"), "n_flux_events")[0])

    fobjs = [
        reffile,
        uproot.open(str(filename) + ".root"),
        tables.open_file(str(filename) + ".hdf5", "r"),
        pd.HDFStore(str(filename) + ".hdf5", "r"),
    ]

    for fobj in fobjs:
        w = GenieWeighter(fobj)

        pdf0 = next(iter(w.surface.spectra.values()))[0].dists[0]
        assert 1 / pdf0.v == approx(global_probability_scale * solid_angle * injection_area, rel=1e-5)

        assert w.get_weight_column("wght") == approx(genie_weight)

        power_law = next(iter(w.surface.spectra.values()))[0].dists[2]
        energy_term = 1 / power_law.pdf(w.get_weight_column("energy"))
        assert energy_term == approx(energy_factor)

        one_weight = w.get_weight_column("wght") * energy_term * solid_angle * injection_area * global_probability_scale
        assert one_weight == approx(wd["OneWeight"], rel=1e-5)

        assert w.get_weights(1) == approx(final_weight, rel=1e-5)

    for fobj in fobjs:
        fobj.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
