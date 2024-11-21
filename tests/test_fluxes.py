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
    simweights.Hoerandel(),
    simweights.Hoerandel5(),
    simweights.Hoerandel_IT(),
    simweights.GaisserHillas(),
    simweights.GaisserH3a(),
    simweights.GaisserH4a(),
    simweights.GaisserH4a_IT(),
    simweights.Honda2004(),
    simweights.TIG1996(),
    simweights.GlobalFitGST(),
    simweights.GlobalFitGST_IT(),
    simweights.GlobalSplineFit(),
    simweights.GlobalSplineFit5Comp(),
    simweights.GlobalSplineFit_IT(),
    simweights.FixedFractionFlux({2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4}),
    simweights.FixedFractionFlux({2212: 0.1, 1000020040: 0.2, 1000080160: 0.3, 1000260560: 0.4}, simweights.GaisserH4a_IT()),
]


@pytest.mark.parametrize("flux", flux_models, ids=[x.__class__.__name__ for x in flux_models])
def test_flux_model(flux, ndarrays_regression):
    # this is the old regression test it can stick around for a bit but will be deleted at a certain point
    for pdgid in flux.pdgids:
        v1 = flux(E, pdgid)
        v2 = np.array(flux_values[flux.__class__.__name__][str(int(pdgid))]) / 1e4
    assert v1 == pytest.approx(v2, rel=1e-13)

    ndarrays_regression.check({pdgid.name: flux(E, pdgid) for pdgid in flux.pdgids}, default_tolerance={"rtol": 1e-13})
    # make sure you get zero for non CR primaries
    assert flux(E, 22) == pytest.approx(0)


gsfmodels = [
    simweights.GlobalSplineFit(),
    simweights.GlobalSplineFit5Comp(),
    simweights.GlobalSplineFit_IT(),
]


@pytest.mark.parametrize("gsf", gsfmodels)
def test_GlobalSplineFit_similar(gsf):
    """
    Test if GlobalSplineFit is similar to to other models within a factor of 0.4,
    mainly to check if the units provided by the GST files match.
    Sum all species before check because different models use different particles.
    This can be not transparent in the file and web-interface.
    If the units mismatch, we should expect at least a deviation of 1000 (one prefix)
    or most likely a mismatch of 10 000 (cm^2 vs m^2).
    """
    for name in ("GlobalFitGST", "GaisserH3a", "Hoerandel5"):
        model = getattr(simweights, name)()
        f_gsf = gsf(*np.meshgrid(E, gsf.pdgids)).sum(axis=0)
        f = model(*np.meshgrid(E, model.pdgids)).sum(axis=0)
        assert f == pytest.approx(f_gsf, rel=0.4)


@pytest.mark.parametrize("gsf", gsfmodels)
def test_GlobalSplineFIt_negative(gsf):
    """
    Since GSF was using a cubic spline, it could oscillate a bit between the interpolated data point.
    This was noticed due to a very minor undershoot below zero for protons at high energies.
    So here we explicitly test that we only return non-negative values.
    """
    test_e = np.geomspace(9.95e10, 10e10, 9)
    flux = gsf(*np.meshgrid(test_e, gsf.pdgids))
    assert np.all(flux >= 0)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
