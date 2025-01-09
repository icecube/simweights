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

from simweights import NuGenWeighter

with contextlib.suppress(ImportError):
    from icecube import dataio

datasets = [
    "Level2_IC86.2016_NuE.020885.000000",
    "Level2_IC86.2016_NuMu.020878.000000",
    "Level2_IC86.2016_NuTau.020895.000000",
    "Level2_IC86.2012_nugen_nue.012646.000000.clsim-base-4.0.5.0.99_eff",
    "Level2_IC86.2012_nugen_nutau.011836.000000.clsim-base-4.0.3.0.99_eff",
    "Level2_IC86.2012_nugen_nutau.011477.000000.clsim-base-4.0.3.0.99_eff",
    "Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff",
    "Level2_nugen_nutau_IC86.2012.011297.000000",
    "Level2_nugen_numu_IC86.2012.011070.000000",
    "Level2_nugen_numu_IC86.2012.011069.000000",
    "Level2_IC86.2012_nugen_NuTau.011065.000001",
    "Level2_nugen_numu_IC86.2012.011029.000000",
    "Level2_IC86.2011_nugen_NuE.010692.000000",
    "Level2_IC86.2011_nugen_NuMu.010634.000000",
]
loaders = [
    pytest.param(lambda f: h5py.File(f"{f}.hdf5", "r"), id="h5py"),
    pytest.param(lambda f: uproot.open(f"{f}.root"), id="uproot"),
    pytest.param(lambda f: tables.open_file(f"{f}.hdf5", "r"), id="pytables"),
    pytest.param(lambda f: pd.HDFStore(f"{f}.hdf5", "r"), id="pandas"),
]

approx = pytest.approx
datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)


def load_reference_values(fname):
    d = {}
    reffile = h5py.File(fname, "r")

    wd = reffile["I3MCWeightDict"]
    d["pdgid"] = wd["PrimaryNeutrinoType"][0]

    d["solid_angle"] = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))
    if "SolidAngle" in wd.dtype.names:
        np.testing.assert_allclose(d["solid_angle"], wd["SolidAngle"])

    if "InjectionAreaCGS" in wd.dtype.names:
        d["injection_area"] = wd["InjectionAreaCGS"]
    if "InjectionAreaNormCGS" in wd.dtype.names:
        d["injection_area"] = wd["InjectionAreaNormCGS"]

    if "TotalWeight" in wd.dtype.names:
        d["total_weight"] = wd["TotalWeight"]
    elif "TotalInteractionProbabilityWeight" in wd.dtype.names:
        d["total_weight"] = wd["TotalInteractionProbabilityWeight"]

    type_weight = wd["TypeWeight"] if "TypeWeight" in wd.dtype.names else 0.5
    d["OneWeight"] = wd["OneWeight"]
    d["OneWeightByN"] = wd["OneWeight"] / (wd["NEvents"] * type_weight)

    return d


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.parametrize("loader", loaders)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(fname, loader):
    filename = Path(datadir) / fname
    ref_values = load_reference_values(f"{filename}.hdf5")

    with loader(filename) as fobj:
        w = NuGenWeighter(fobj, nfiles=1)

        event_weight = w.get_weight_column("event_weight")
        assert event_weight == approx(ref_values["total_weight"])

        cylinder = w.surface.spectra[ref_values["pdgid"]][0].dists[2]
        proj_area = cylinder.projected_area(w.get_weight_column("cos_zen"))
        assert proj_area == approx(ref_values["injection_area"])

        sw_etendue = 1 / cylinder.pdf(w.get_weight_column("cos_zen"))
        assert sw_etendue == approx(ref_values["solid_angle"] * ref_values["injection_area"], rel=1e-5)

        power_law = w.surface.spectra[ref_values["pdgid"]][0].dists[1]
        energy_factor = 1 / power_law.pdf(w.get_weight_column("energy"))
        one_weight = (
            w.get_weight_column("event_weight") * energy_factor * ref_values["solid_angle"] * ref_values["injection_area"]
        )
        assert one_weight == approx(ref_values["OneWeight"])
        assert w.get_weights(1) == approx(ref_values["OneWeightByN"], rel=1e-5)


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
@pytest.mark.skipif("dataio" not in globals(), reason="not in icetray environment")
def test_dataset_i3file(fname):
    d = load_reference_values(Path(datadir) / (fname + ".hdf5"))

    i = 0
    f = dataio.I3File(str(Path(datadir) / (fname + ".i3.zst")))
    while f.more():
        frame = f.pop_frame()
        if frame.Stop != frame.Physics:
            continue
        w = NuGenWeighter(frame, nfiles=1)
        event_weight = w.get_weight_column("event_weight")
        assert event_weight == approx(d["total_weight"][i])

        pdgid = int(frame["I3MCWeightDict"]["PrimaryNeutrinoType"])
        cylinder = w.surface.spectra[pdgid][0].dists[2]
        proj_area = cylinder.projected_area(w.get_weight_column("cos_zen"))
        assert proj_area == approx(d["injection_area"][i])

        sw_etendue = 1 / cylinder.pdf(w.get_weight_column("cos_zen"))
        assert sw_etendue == approx(d["solid_angle"][i] * d["injection_area"][i], rel=1e-5)

        power_law = w.surface.spectra[pdgid][0].dists[1]
        energy_factor = 1 / power_law.pdf(w.get_weight_column("energy"))
        one_weight = w.get_weight_column("event_weight") * energy_factor * d["solid_angle"][i] * d["injection_area"][i]

        assert one_weight == approx(d["OneWeight"][i])
        assert d["OneWeightByN"][i] == approx(w.get_weights(1), rel=1e-5)
        i += 1
    f.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
