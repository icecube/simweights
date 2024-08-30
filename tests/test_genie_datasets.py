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

datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)
datasets = [
    "upgrade_genie_step3_141828_000000",
    "GENIE_NuMu_IceCubeUpgrade_v58.22590.000000",
    "genie_numu_volume_scaling",
]
approx = pytest.approx


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(fname):
    filename = Path(datadir) / fname
    reffile = h5py.File(str(filename) + ".hdf5", "r")
    info = reffile["I3GenieInfo"][0]
    wd = reffile["I3MCWeightDict"]

    n_flux_events = info["n_flux_events"]
    primary_type = info["primary_type"]
    cylinder_radius = info["cylinder_radius"]
    min_zenith = info["min_zenith"]
    max_zenith = info["max_zenith"]
    min_energy = info["min_energy"]
    max_energy = info["max_energy"]
    power_law_index = info["power_law_index"]
    global_probability_scale = info["global_probability_scale"]
    muon_volume_scaling = info["muon_volume_scaling"]

    if "PrimaryNeutrinoType" in wd.dtype.names:
        assert np.all(wd["PrimaryNeutrinoType"] == primary_type)
    assert wd["InjectionSurfaceR"] == approx(cylinder_radius)
    assert wd["MinZenith"] == approx(min_zenith)
    assert wd["MaxZenith"] == approx(max_zenith)
    assert 10 ** wd["MinEnergyLog"] == approx(min_energy)
    assert 10 ** wd["MaxEnergyLog"] == approx(max_energy)
    assert wd["PowerLawIndex"] == approx(power_law_index)
    assert wd["GlobalProbabilityScale"] == approx(global_probability_scale)
    if "MuonVolumeScaling" in wd.dtype.names:
        assert wd["MuonVolumeScaling"] == approx(muon_volume_scaling)

    solid_angle = 2 * np.pi * (np.cos(min_zenith) - np.cos(max_zenith))
    injection_area = np.pi * (cylinder_radius * 1e2) ** 2

    if power_law_index == 1:
        energy_integral = np.log(max_energy / min_energy)
    else:
        energy_integral = (max_energy ** (1 - power_law_index) - min_energy ** (1 - power_law_index)) / (1 - power_law_index)
    energy_factor = 1 / (wd["PrimaryNeutrinoEnergy"] ** (-power_law_index) / energy_integral)

    one_weight = wd["TotalInteractionProbabilityWeight"] * energy_factor * solid_angle * injection_area
    np.testing.assert_allclose(one_weight, wd["OneWeight"])
    final_weight = wd["OneWeight"] / n_flux_events

    fobjs = [
        reffile,
        uproot.open(str(filename) + ".root"),
        tables.open_file(str(filename) + ".hdf5", "r"),
        pd.HDFStore(str(filename) + ".hdf5", "r"),
    ]

    for fobj in fobjs:
        w = GenieWeighter(fobj)

        gprob, _, _, edist = next(iter(w.surface.spectra.values()))[0].dists
        energy_term = 1 / edist.pdf(w.get_weight_column("energy"))

        assert gprob.v == approx(1 / solid_angle / injection_area / global_probability_scale)
        assert w.get_weight_column("wght") == approx(wd["GENIEWeight"])
        assert energy_term == approx(energy_factor)

        vol_scale = w.get_weight_column("volscale")
        one_weight = (
            w.get_weight_column("wght") * energy_term * solid_angle * injection_area * global_probability_scale * vol_scale
        )
        assert one_weight == approx(wd["OneWeight"], rel=1e-5)

        assert w.get_weights(1) == approx(final_weight, rel=1e-5)

    for fobj in fobjs:
        fobj.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
