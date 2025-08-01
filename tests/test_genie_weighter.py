#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
import sys

import numpy as np
import pytest
from pytest import approx

import simweights

with contextlib.suppress(ImportError):
    from icecube import dataclasses, icetray, simclasses

info_dtype = [
    ("primary_type", np.int32),
    ("n_flux_events", np.int32),
    ("global_probability_scale", np.float64),
    ("cylinder_radius", np.float64),
    ("min_zenith", np.float64),
    ("max_zenith", np.float64),
    ("min_energy", np.float64),
    ("max_energy", np.float64),
    ("power_law_index", np.float64),
]

result_dtype = [("neu", np.int32), ("pzv", np.float64), ("Ev", np.float64), ("wght", np.float64)]


def test_genie_repr():
    c1 = simweights.CircleInjector(300, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    g = simweights.GenieSurface(-14, 1000, 0.3333, c1, p1)
    GenieSurface = simweights.GenieSurface  # noqa: F841
    CircleInjector = simweights.CircleInjector  # noqa: F841
    PowerLaw = simweights.PowerLaw  # noqa: F841
    assert eval(repr(g)) == g


@pytest.mark.parametrize("event_weight", (1e-6, 1e-3, 1))
@pytest.mark.parametrize("nfiles", (1, 5, 50))
@pytest.mark.parametrize("include_volscale", (True, False))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
def test_genie_reader_weighter(event_weight, nfiles, include_volscale, flux):
    nevents = 10000
    coszen = 0.7
    pdgid = 12
    c1 = simweights.CircleInjector(300, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)

    result_dtype = [("neu", np.int32), ("pzv", np.float64), ("Ev", np.float64), ("wght", np.float64)]
    if include_volscale:
        result_dtype.append(("volscale", np.float64))

    weight = np.zeros(nevents, dtype=result_dtype)
    weight["neu"] = pdgid
    weight["pzv"] = coszen
    weight["Ev"] = p1.ppf(np.linspace(0, 1, nevents))
    weight["wght"] = event_weight

    if include_volscale:
        weight["volscale"] = 1

    rows = nfiles * [
        (
            pdgid,
            nevents,
            1,
            c1.radius,
            np.arccos(c1.cos_zen_max),
            np.arccos(c1.cos_zen_min),
            p1.a,
            p1.b,
            p1.g,
        ),
    ]
    info = np.array(rows, dtype=info_dtype)
    d = {"I3GenieResult": weight, "I3GenieInfo": info}

    wobj = simweights.GenieWeighter(d)
    w = wobj.get_weights(flux)
    np.testing.assert_allclose(
        w.sum(),
        flux * event_weight * c1.etendue * p1.integral / nfiles,
    )
    E = d["I3GenieResult"]["Ev"]
    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    np.testing.assert_allclose(y, flux * event_weight * Ewidth * c1.etendue / nfiles, 5e-3)

    with pytest.raises(RuntimeError):
        simweights.GenieWeighter(d, nfiles=10)

    with pytest.raises(TypeError):
        simweights.GenieWeighter({"I3CorsikaWeight": weight})

    with pytest.raises(KeyError):
        simweights.GenieWeighter({"I3GenieResult": weight})


@pytest.mark.parametrize("event_weight", (1e-6, 1e-3, 1))
@pytest.mark.parametrize("volscale", (1, 2, 3))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
@pytest.mark.skipif("dataclasses" not in globals(), reason="Not in an IceTray environment")
def test_genie_reader_weighter_i3file(event_weight, volscale, flux):
    nevents = 10000
    coszen = 0.7
    pdgid = 12
    energy = 5e3
    c1 = simweights.CircleInjector(300, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)

    weight = simclasses.I3GenieResult()
    weight.neu = pdgid
    weight.pzv = coszen
    weight.Ev = energy
    weight.wght = event_weight
    weight.volscale = volscale

    info = simclasses.I3GenieInfo()
    info.primary_type = dataclasses.I3Particle.ParticleType(pdgid)
    info.n_flux_events = nevents
    info.global_probability_scale = 1
    info.cylinder_radius = c1.radius
    info.min_zenith = np.arccos(c1.cos_zen_max)
    info.max_zenith = np.arccos(c1.cos_zen_min)
    info.min_energy = p1.a
    info.max_energy = p1.b
    info.power_law_index = p1.g

    frame = icetray.I3Frame()
    frame["I3GenieResult"] = weight
    frame["I3GenieInfo"] = info

    w = simweights.GenieWeighter(frame).get_weights(flux)
    assert w == approx(flux * volscale * event_weight / c1.pdf(coszen) / p1.pdf(energy) / nevents)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
