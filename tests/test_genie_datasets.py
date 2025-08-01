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
from pytest import approx

from simweights import GenieWeighter

with contextlib.suppress(ImportError):
    from icecube import dataio, simclasses  # noqa: F401

datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
genie_reader_datasets = [
    "upgrade_genie_step3_141828_000000",
    "GENIE_NuMu_IceCubeUpgrade_v58.22590.000000",
    "genie_numu_volume_scaling",
]

loaders = [
    pytest.param(lambda f: h5py.File(str(f) + ".hdf5", "r"), id="h5py"),
    pytest.param(lambda f: uproot.open(str(f) + ".root"), id="uproot"),
    pytest.param(lambda f: tables.open_file(str(f) + ".hdf5", "r"), id="pytables"),
    pytest.param(lambda f: pd.HDFStore(str(f) + ".hdf5", "r"), id="pandas"),
]


def load_reference_values(fname):
    ref = {}
    reffile = h5py.File(str(fname) + ".hdf5", "r")
    info = reffile["I3GenieInfo"][0]
    wd = reffile["I3MCWeightDict"]
    ref["GENIEWeight"] = wd["GENIEWeight"]

    n_flux_events = info["n_flux_events"]
    primary_type = info["primary_type"]
    cylinder_radius = info["cylinder_radius"]
    min_zenith = info["min_zenith"]
    max_zenith = info["max_zenith"]
    min_energy = info["min_energy"]
    max_energy = info["max_energy"]
    power_law_index = info["power_law_index"]
    ref["global_probability_scale"] = info["global_probability_scale"]
    muon_volume_scaling = info["muon_volume_scaling"]

    if "PrimaryNeutrinoType" in wd.dtype.names:
        assert np.all(wd["PrimaryNeutrinoType"] == primary_type)
    assert wd["InjectionSurfaceR"] == approx(cylinder_radius)
    assert wd["MinZenith"] == approx(min_zenith)
    assert wd["MaxZenith"] == approx(max_zenith)
    assert 10 ** wd["MinEnergyLog"] == approx(min_energy)
    assert 10 ** wd["MaxEnergyLog"] == approx(max_energy)
    assert wd["PowerLawIndex"] == approx(power_law_index)
    assert wd["GlobalProbabilityScale"] == approx(ref["global_probability_scale"])
    if "MuonVolumeScaling" in wd.dtype.names:
        assert wd["MuonVolumeScaling"] == approx(muon_volume_scaling)

    ref["pdgid"] = primary_type
    ref["solid_angle"] = 2 * np.pi * (np.cos(min_zenith) - np.cos(max_zenith))
    ref["injection_area"] = np.pi * (cylinder_radius * 1e2) ** 2

    if power_law_index == 1:
        energy_integral = np.log(max_energy / min_energy)
    else:
        energy_integral = (max_energy ** (1 - power_law_index) - min_energy ** (1 - power_law_index)) / (1 - power_law_index)
    ref["energy_factor"] = 1 / (wd["PrimaryNeutrinoEnergy"] ** (-power_law_index) / energy_integral)

    ref["one_weight"] = (
        wd["TotalInteractionProbabilityWeight"] * ref["energy_factor"] * ref["solid_angle"] * ref["injection_area"]
    )
    np.testing.assert_allclose(ref["one_weight"], wd["OneWeight"])
    ref["final_weight"] = wd["OneWeight"] / n_flux_events
    return ref


@pytest.mark.parametrize("fname", genie_reader_datasets)
@pytest.mark.parametrize("loader", loaders)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_genie_reader_dataset(fname, loader):
    filename = Path(datadir) / fname
    ref = load_reference_values(filename)

    fobj = loader(filename)
    w = GenieWeighter(fobj)
    energy_term = 1 / w.surface.components[ref["pdgid"]][0].power_law.pdf(w.get_weight_column("energy"))

    assert w.surface.components[ref["pdgid"]][0].global_probability_scale == approx(ref["global_probability_scale"])
    assert w.get_weight_column("wght") == approx(ref["GENIEWeight"])
    assert energy_term == approx(ref["energy_factor"])

    vol_scale = w.get_weight_column("volscale")
    one_weight = (
        w.get_weight_column("wght")
        * energy_term
        * ref["solid_angle"]
        * ref["injection_area"]
        * ref["global_probability_scale"]
        * vol_scale
    )
    assert one_weight == approx(ref["one_weight"], rel=1e-5)
    assert w.get_weights(1) == approx(ref["final_weight"], rel=1e-5)

    fobj.close()


@pytest.mark.parametrize("fname", genie_reader_datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
@pytest.mark.skipif("dataio" not in globals(), reason="Not in an IceTray environment")
def test_dataset_i3file(fname):
    filename = Path(datadir) / fname
    ref = load_reference_values(filename)

    i = 0
    f = dataio.I3File(str(Path(datadir) / (fname + ".i3.zst")))
    while f.more():
        frame = f.pop_frame()
        if frame.Stop != frame.Physics:
            continue
        w = GenieWeighter(frame)

        gprob, _, _, edist = next(iter(w.surface.spectra.values()))[0].dists
        energy_term = 1 / edist.pdf(w.get_weight_column("energy"))

        assert gprob.v == approx(1 / ref["solid_angle"] / ref["injection_area"] / ref["global_probability_scale"])
        assert w.get_weight_column("wght") == approx(ref["GENIEWeight"][i])
        assert energy_term == approx(ref["energy_factor"][i])

        vol_scale = w.get_weight_column("volscale")
        one_weight = (
            w.get_weight_column("wght")
            * energy_term
            * ref["solid_angle"]
            * ref["injection_area"]
            * ref["global_probability_scale"]
            * vol_scale
        )
        assert one_weight == approx(ref["one_weight"][i], rel=1e-5)

        assert w.get_weights(1) == approx(ref["final_weight"][i], rel=1e-5)
        i += 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
