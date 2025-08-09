#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
import sys

import numpy as np
import pytest
from pytest import approx
from scipy.interpolate import interp1d

import simweights
from simweights import CorsikaWeighter

with contextlib.suppress(ImportError):
    from icecube import dataclasses, icetray, simclasses

info_dtype = [
    ("n_events", np.int32),
    ("primary_type", np.int32),
    ("oversampling", np.int32),
    ("cylinder_height", np.float64),
    ("cylinder_radius", np.float64),
    ("min_zenith", np.float64),
    ("max_zenith", np.float64),
    ("min_energy", np.float64),
    ("max_energy", np.float64),
    ("power_law_index", np.float64),
]

primary_dtype = [("type", np.int32), ("energy", np.float64), ("zenith", np.float64)]

weight_dtype = [
    ("ParticleType", np.int32),
    ("CylinderLength", np.float64),
    ("CylinderRadius", np.float64),
    ("ThetaMin", np.float64),
    ("ThetaMax", np.float64),
    ("OverSampling", np.float64),
    ("Weight", np.float64),
    ("NEvents", np.float64),
    ("EnergyPrimaryMin", np.float64),
    ("EnergyPrimaryMax", np.float64),
    ("PrimarySpectralIndex", np.float64),
    ("PrimaryEnergy", np.float64),
    ("PrimaryType", np.float64),
]


def get_cos_zenith_dist(c, N):
    cz = np.linspace(c.cos_zen_min, c.cos_zen_max, 1000)
    cdf = (c._diff_etendue(cz) - c._diff_etendue(c.cos_zen_min)) / (
        c._diff_etendue(c.cos_zen_max) - c._diff_etendue(c.cos_zen_min)
    )
    cf = interp1d(cdf, cz)
    p = np.linspace(0, 1, N)
    return cf(p)


def make_corsika_data(pdgid, nevents, c, p):
    weight = np.zeros(nevents, dtype=weight_dtype)
    weight["ParticleType"] = pdgid
    weight["CylinderLength"] = c.length
    weight["CylinderRadius"] = c.radius
    weight["ThetaMin"] = np.arccos(c.cos_zen_max)
    weight["ThetaMax"] = np.arccos(c.cos_zen_min)
    weight["OverSampling"] = 1
    weight["Weight"] = 1
    weight["NEvents"] = nevents
    weight["EnergyPrimaryMin"] = p.a
    weight["EnergyPrimaryMax"] = p.b
    weight["PrimarySpectralIndex"] = p.g
    weight["PrimaryEnergy"] = p.ppf(np.linspace(0, 1, nevents))
    weight["PrimaryType"] = pdgid
    return weight


@pytest.mark.parametrize("oversampling", (1, 5, 50))
@pytest.mark.parametrize("nfiles", (1, 10, 100))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
def test_old_corsika(oversampling, nfiles, flux):
    c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    d = make_corsika_data(2212, 10000, c1, p1)

    d["OverSampling"] = oversampling
    wobj = CorsikaWeighter({"CorsikaWeightMap":d}, nfiles=nfiles)

    w = wobj.get_weights(flux)
    np.testing.assert_allclose(
        w.sum(),
        flux * c1.etendue * p1.integral / nfiles / oversampling,
    )
    E = d["PrimaryEnergy"]
    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / nfiles / oversampling, 5e-3)

    with pytest.raises(RuntimeError):
        CorsikaWeighter({"CorsikaWeigthMap":d})

    with pytest.raises(TypeError):
        CorsikaWeighter({"CorsikaWeigthMap":d}, nfiles=object())

    with pytest.raises(RuntimeError):
        x = {"CorsikaWeightMap": {"ParticleType": []}, "PolyplopiaPrimary": {}}
        CorsikaWeighter(x, nfiles=1)


@pytest.mark.parametrize("oversampling", (1, 5, 50))
@pytest.mark.parametrize("nfiles", (1, 10, 100))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
def test_sframe_corsika(oversampling, nfiles, flux):
    c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    d = {"CorsikaWeightMap": make_corsika_data(2212, 10000, c1, p1)}
    rows = nfiles * [
        (
            10000,
            2212,
            oversampling,
            c1.length,
            c1.radius,
            np.arccos(c1.cos_zen_max),
            np.arccos(c1.cos_zen_min),
            p1.a,
            p1.b,
            p1.g,
        ),
    ]
    d["I3CorsikaInfo"] = np.array(rows, dtype=info_dtype)
    wobj = CorsikaWeighter(d)

    w = wobj.get_weights(flux)
    np.testing.assert_allclose(
        w.sum(),
        flux * c1.etendue * p1.integral / nfiles / oversampling,
    )
    E = d["CorsikaWeightMap"]["PrimaryEnergy"]
    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    np.testing.assert_allclose(y, flux * Ewidth * c1.etendue / nfiles / oversampling, 5e-3)

    with pytest.warns(UserWarning):
        CorsikaWeighter(d, nfiles=10)


@pytest.mark.parametrize("event_weight", (1e-6, 1e-3, 1))
@pytest.mark.parametrize("nfiles", (1, 5, 50))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
def test_triggered_corsika(event_weight, nfiles, flux):
    weight_dtype = [
        ("type", np.int32),
        ("energy", np.float64),
        ("zenith", np.float64),
        ("weight", np.float64),
    ]
    info_dtype = [
        ("primary_type", np.int32),
        ("n_events", np.int32),
        ("cylinder_height", np.float64),
        ("cylinder_radius", np.float64),
        ("min_zenith", np.float64),
        ("max_zenith", np.float64),
        ("min_energy", np.float64),
        ("max_energy", np.float64),
        ("power_law_index", np.float64),
    ]

    nevents = 10000
    c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    d = make_corsika_data(2212, 10000, c1, p1)
    weight = np.zeros(nevents, dtype=weight_dtype)
    weight["type"] = 2212
    weight["zenith"] = np.arccos(get_cos_zenith_dist(c1, nevents))
    weight["energy"] = p1.ppf(np.linspace(0, 1, nevents))

    weight["weight"] = event_weight

    rows = nfiles * [
        (
            2212,
            nevents,
            c1.length,
            c1.radius,
            np.arccos(c1.cos_zen_max),
            np.arccos(c1.cos_zen_min),
            p1.a,
            p1.b,
            p1.g,
        ),
    ]
    info = np.array(rows, dtype=info_dtype)
    d = {"I3CorsikaWeight": weight, "I3PrimaryInjectorInfo": info}

    wobj = CorsikaWeighter(d)
    w = wobj.get_weights(flux)
    np.testing.assert_allclose(
        w.sum(),
        flux * event_weight * c1.etendue * p1.integral / nfiles,
    )
    E = d["I3CorsikaWeight"]["energy"]
    y, x = np.histogram(E, weights=w, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    np.testing.assert_allclose(y, flux * event_weight * Ewidth * c1.etendue / nfiles, 5e-3)

    with pytest.raises(RuntimeError):
        CorsikaWeighter(d, nfiles=10)

    with pytest.raises(RuntimeError):
        CorsikaWeighter({"I3CorsikaWeight": weight})


@pytest.mark.parametrize("oversampling", (1, 5, 50))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_old_corsika_i3file(oversampling, flux):
    c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    cwm = make_corsika_data(2212, 1, c1, p1)
    cwm["OverSampling"] = oversampling
    wm = dataclasses.I3MapStringDouble({k: float(cwm[k][0]) for k in cwm.dtype.names})
    pp = dataclasses.I3Particle()
    pp.type = pp.ParticleType(cwm["PrimaryType"][0])
    pp.energy = cwm["PrimaryEnergy"][0]
    frame = icetray.I3Frame()
    frame["CorsikaWeightMap"] = wm
    frame["PolyplopiaPrimary"] = pp
    wobj = CorsikaWeighter(frame, nfiles=1)
    w = wobj.get_weights(flux)
    assert w == approx(flux * c1.etendue / p1.pdf(pp.energy) / oversampling)


@pytest.mark.parametrize("oversampling", (1, 10, 100, 1000))
@pytest.mark.parametrize("n_events", (1, 10, 100))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_sframe_corsika_i3files(oversampling, n_events, flux):
    c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    d = make_corsika_data(2212, 1, c1, p1)

    info = simclasses.I3CorsikaInfo()
    info.n_events = n_events
    info.primary_type = dataclasses.I3Particle.ParticleType(2212)
    info.oversampling = oversampling
    info.cylinder_height = c1.length
    info.cylinder_radius = c1.radius
    info.min_zenith = np.arccos(c1.cos_zen_max)
    info.max_zenith = np.arccos(c1.cos_zen_min)
    info.power_law_index = p1.g
    info.min_energy = p1.a
    info.max_energy = p1.b

    wm = dataclasses.I3MapStringDouble()
    wm["PrimaryEnergy"] = d["PrimaryEnergy"][0]
    wm["PrimaryType"] = d["PrimaryType"][0]
    wm["Weight"]  = 1.

    frame = icetray.I3Frame()
    frame["I3CorsikaInfo"] = info
    frame["CorsikaWeightMap"]  = wm
    w = CorsikaWeighter(frame).get_weights(flux)
    assert w == approx(flux * c1.etendue / p1.pdf(wm["PrimaryEnergy"]) / n_events / oversampling)


@pytest.mark.parametrize("event_weight", (1e-6, 1e-3, 1))
@pytest.mark.parametrize("nevents", (1, 5, 50))
@pytest.mark.parametrize("flux", (0.1, 1, 10))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_triggered_corsika_i3file(event_weight, nevents, flux):
    c1 = simweights.NaturalRateCylinder(1200, 600, 0, 1)
    p1 = simweights.PowerLaw(0, 1e3, 1e4)
    cwm = make_corsika_data(2212, 1, c1, p1)

    primary = dataclasses.I3Particle()
    primary.type = primary.ParticleType(2212)
    primary.energy = cwm["PrimaryEnergy"][0]
    weight = simclasses.I3CorsikaWeight()
    weight.primary = primary
    weight.weight = event_weight

    # you can't set values for I3PrimaryInjectorInfo in python so lets just fake it
    info = dataclasses.I3MapStringDouble(
        {
            "n_events": nevents,
            "primary_type": 2212,
            "cylinder_height": c1.length,
            "cylinder_radius": c1.radius,
            "min_zenith": np.arccos(c1.cos_zen_max),
            "max_zenith": np.arccos(c1.cos_zen_min),
            "min_energy": p1.a,
            "max_energy": p1.b,
            "power_law_index": p1.g,
        }
    )
    frame = icetray.I3Frame()
    frame["I3CorsikaWeight"] = weight
    frame["I3PrimaryInjectorInfo"] = info

    w = CorsikaWeighter(frame).get_weights(flux)
    assert w == approx(flux * event_weight * c1.etendue * p1.integral / nevents)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
