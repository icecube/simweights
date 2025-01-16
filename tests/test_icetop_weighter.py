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
    ("n_events", np.int32),
    ("sampling_radius", np.float64),
    ("min_zenith", np.float64),
    ("max_zenith", np.float64),
    ("min_energy", np.float64),
    ("max_energy", np.float64),
    ("power_law_index", np.float64),
]

particle_dtype = [("type", np.int32), ("energy", np.float64), ("zenith", np.float64)]


@pytest.mark.parametrize("nfiles", (1, 5, 50))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
def test_icetop_corsika(nfiles, flux):
    nevents = 10000
    pdgid = 12
    c1 = simweights.NaturalRateCylinder(0, 300, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)

    weight = np.zeros(nevents, dtype=particle_dtype)
    weight["type"] = pdgid
    weight["energy"] = p1.ppf(np.linspace(0, 1, nevents))
    weight["zenith"] = np.arccos(np.linspace(c1.cos_zen_min, c1.cos_zen_max, nevents))

    rows = nfiles * [
        (
            pdgid,
            nevents,
            c1.radius,
            np.arccos(c1.cos_zen_max),
            np.arccos(c1.cos_zen_min),
            p1.a,
            p1.b,
            p1.g,
        ),
    ]
    info = np.array(rows, dtype=info_dtype)
    d = {"MCPrimary": weight, "I3TopInjectorInfo": info}

    wobj = simweights.IceTopWeighter(d)
    w = wobj.get_weights(flux)
    np.testing.assert_allclose(w.sum(), flux * c1.etendue * p1.integral / nfiles)
    E = d["MCPrimary"]["energy"]
    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / nfiles, 5e-3)

    with pytest.raises(TypeError):
        simweights.IceTopWeighter(d, nfiles=10)

    with pytest.raises(KeyError):
        simweights.IceTopWeighter({"MCParticle": weight})

    with pytest.raises(KeyError):
        simweights.IceTopWeighter({"I3TopInjectorInfo": info})


@pytest.mark.parametrize("nevents", (1000, 10000, 100000))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_icetop_corsika_i3files(nevents, flux):
    pdgid = 12
    c1 = simweights.NaturalRateCylinder(0, 300, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)

    info = simclasses.I3TopInjectorInfo()
    info.n_events = nevents
    info.primary_type = dataclasses.I3Particle.ParticleType(pdgid)
    info.sampling_radius = c1.radius
    info.min_zenith = np.arccos(c1.cos_zen_max)
    info.max_zenith = np.arccos(c1.cos_zen_min)
    info.power_law_index = p1.g
    info.min_energy = p1.a
    info.max_energy = p1.b

    primary = dataclasses.I3Particle()
    primary.type = primary.ParticleType(pdgid)
    primary.energy = p1.a
    primary.dir = dataclasses.I3Direction(np.arccos(c1.cos_zen_max), 0)

    frame = icetray.I3Frame()
    frame["I3TopInjectorInfo"] = info
    frame["MCPrimary"] = primary
    w = simweights.IceTopWeighter(frame).get_weights(flux)
    assert w == approx(flux / c1.pdf(c1.cos_zen_max) / p1.pdf(primary.energy) / nevents)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
