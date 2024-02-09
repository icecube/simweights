#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import simweights

with (Path(__file__).parent / "flux_values.json").open() as f:
    flux_values = json.load(f)
E = np.logspace(2, 10, 9)

flux_models = [
    ("Hoerandel", ()),
    ("Hoerandel5", ()),
    ("Hoerandel_IT", ()),
    ("GaisserHillas", ()),
    ("GaisserH3a", ()),
    ("GaisserH4a", ()),
    ("GaisserH4a_IT", ()),
    ("Honda2004", ()),
    ("TIG1996", ()),
    ("GlobalFitGST", ()),
    ("GlobalFitGST_IT", ()),
    ("GlobalSplineFit", ()),
    ("GlobalSplineFit5Comp", ()),
    ("GlobalSplineFit_IT", ()),
    ("FixedFractionFlux", ({2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4},)),
    ("FixedFractionFlux", ({2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4}, simweights.GaisserH4a_IT())),
]


@pytest.mark.parametrize(("name", "args"), flux_models)
# @pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
def test_flux_model(name, args):
    flux = getattr(simweights._fluxes, name)(*args)
    for pdgid in flux.pdgids:
        v1 = flux(E, pdgid)
        v2 = np.array(flux_values[name][str(int(pdgid))]) / 1e4
    # make sure you get zero for non CR primaries
    assert v1 == pytest.approx(v2, rel=1e-13)


gsfmodels = [
    simweights.GlobalSplineFit(),
    simweights.GlobalSplineFit5Comp(),
    simweights.GlobalSplineFit_IT(),
]


@pytest.mark.parametrize("gsf", gsfmodels)
def test_GlobalSplineFit5Comp_similar(gsf):
    """
    Test if GlobalSplineFit is similar to to other models within a factor of 500,
    mainly to check if the units provided by the GST files match.
    This can be not transparent in the file and web-interface.
    If the units mismatch, we should expect at least a deviation of 1000 (one prefix)
    or most likely a mismatch of 10 000 (cm^2 vs m^2).
    """
    for name in ("GlobalFitGST", "GaisserH3a", "Hoerandel5"):
        model = getattr(simweights, name)()
        f_gsf = gsf(*np.meshgrid(E, gsf.pdgids)).sum(axis=0)
        f = model(*np.meshgrid(E, model.pdgids)).sum(axis=0)
        assert f == pytest.approx(f_gsf, rel=0.4)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
