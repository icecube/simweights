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

from simweights import GenieWeighter

datasets = [
    "genie-icetray.140000A_000000.hdf5",
    "genie-icetray.140000B_000000.hdf5",
    "genie-icetray.140000C_000000.hdf5",
    "genie-icetray.140000D_000000.hdf5",
    "level2_genie-icetray.140000_000000.hdf5",
]
approx = pytest.approx
datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)


@pytest.mark.parametrize("fname", datasets)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_dataset(fname):
    filename = Path(datadir) / fname
    reffile = h5py.File(str(filename), "r")

    wd = reffile["I3MCWeightDict"]
    grd = reffile["I3GENIEResultDict"]
    pdgid = grd["neu"]
    emin, emax = 10 ** wd["MinEnergyLog"], 10 ** wd["MaxEnergyLog"]

    solid_angle = 2 * np.pi * (np.cos(wd["MinZenith"]) - np.cos(wd["MaxZenith"]))
    injection_area_cm = 1e4 * np.pi * wd["InjectionSurfaceR"] ** 2
    total_weight = wd["TotalInteractionProbabilityWeight"]

    type_weight = np.empty_like(wd["OneWeight"])
    type_weight[pdgid > 0] = 0.7
    type_weight[pdgid < 0] = 0.3
    w0 = wd["OneWeight"] / (wd["NEvents"] * type_weight)

    fobjs = [
        tables.open_file(str(filename), "r"),
        pd.HDFStore(str(filename), "r"),
    ]

    for fobj in fobjs:
        w = GenieWeighter(fobj, nfiles=1)

        event_weight = w.get_weight_column("wght")
        assert event_weight == approx(total_weight)

        for particle in np.unique(pdgid):
            for spectrum in w.surface.spectra[particle]:
                power_min, power_max = spectrum.dists[1].a, spectrum.dists[1].b
                event_mask = (pdgid == particle) & (emin == power_min) & (emax == power_max)

                power_law = spectrum.dists[1]
                energy_factor = 1 / power_law.pdf(w.get_weight_column("energy"))

                one_weight = (
                    w.get_weight_column("wght")[event_mask]
                    * energy_factor[event_mask]
                    * solid_angle[event_mask]
                    * injection_area_cm[event_mask]
                )

                assert one_weight == approx(wd["OneWeight"][event_mask])

        assert w0 == approx(w.get_weights(1), rel=1e-5)
        fobj.close()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
