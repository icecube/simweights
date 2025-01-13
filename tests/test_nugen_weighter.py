#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from simweights import CircleInjector, NuGenWeighter, PowerLaw, UniformSolidAngleCylinder

with contextlib.suppress(ImportError):
    from icecube import dataclasses, icetray

base_keys = [
    "NEvents",
    "MinZenith",
    "MaxZenith",
    "PowerLawIndex",
    "MinEnergyLog",
    "MaxEnergyLog",
    "PrimaryNeutrinoType",
    "PrimaryNeutrinoZenith",
    "PrimaryNeutrinoEnergy",
]


def make_new_table(pdgid, nevents, spatial, spectrum):
    dtype = [(k, float) for k in base_keys]
    weight = np.zeros(nevents, dtype=dtype)
    weight["PrimaryNeutrinoType"] = pdgid
    weight["NEvents"] = nevents
    weight["MinZenith"] = np.arccos(spatial.cos_zen_max)
    weight["MaxZenith"] = np.arccos(spatial.cos_zen_min)
    weight["PowerLawIndex"] = spectrum.g
    weight["MinEnergyLog"] = np.log10(spectrum.a)
    weight["MaxEnergyLog"] = np.log10(spectrum.b)
    weight["PrimaryNeutrinoZenith"] = np.arccos(
        np.linspace(spatial.cos_zen_max, spatial.cos_zen_min, nevents),
    )
    weight["PrimaryNeutrinoEnergy"] = spectrum.ppf(np.linspace(0, 1, nevents))
    return weight


@pytest.mark.parametrize("weight", (0.1, 1, 10))
@pytest.mark.parametrize("nfiles", (1, 10, 100))
@pytest.mark.parametrize("flux", (1e-6, 1, 1e6))
def test_nugen_energy_post_V6(weight, nfiles, flux):
    p1 = PowerLaw(0, 1e3, 1e4)
    c1 = UniformSolidAngleCylinder(200, 100, 0, 0.001)
    t1 = pd.DataFrame(make_new_table(14, 10000, c1, p1))
    t1["CylinderHeight"] = c1.length
    t1["CylinderRadius"] = c1.radius
    t1["TypeWeight"] = 0.5
    t1["TotalWeight"] = weight
    f1 = {"I3MCWeightDict": t1}
    wf = NuGenWeighter(f1, nfiles=nfiles)

    w1 = wf.get_weights(flux)
    assert_allclose(
        w1[1:],
        flux
        * weight
        / (
            t1["TypeWeight"]
            * t1["NEvents"]
            * nfiles
            * c1.pdf(np.cos(t1["PrimaryNeutrinoZenith"]))
            * p1.pdf(t1["PrimaryNeutrinoEnergy"])
        )[1:],
    )
    assert w1.sum() == approx(2 * weight * flux * p1.integral * c1.etendue / nfiles)
    E = t1["PrimaryNeutrinoEnergy"]
    y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    assert y == approx(2 * weight * flux * Ewidth * c1.etendue / nfiles, 6e-3)


@pytest.mark.parametrize("weight", (0.1, 1, 10))
@pytest.mark.parametrize("nfiles", (1, 10, 100))
@pytest.mark.parametrize("flux", (1e-6, 1, 1e6))
def test_nugen_energy_pre_V6(weight, nfiles, flux):
    p1 = PowerLaw(0, 1e3, 1e4)
    c1 = UniformSolidAngleCylinder(1900, 950, 0, 0.001)
    t1 = pd.DataFrame(make_new_table(14, 10000, c1, p1))
    t1["TotalWeight"] = weight
    t1["InjectionSurfaceR"] = -1
    t1["TypeWeight"] = 0.5
    f1 = {"I3MCWeightDict": t1}
    wf = NuGenWeighter(f1, nfiles=nfiles)
    w1 = wf.get_weights(flux)
    assert_allclose(
        w1[1:],
        flux
        * weight
        / (
            t1["TypeWeight"]
            * t1["NEvents"]
            * nfiles
            * c1.pdf(np.cos(t1["PrimaryNeutrinoZenith"]))
            * p1.pdf(t1["PrimaryNeutrinoEnergy"])
        )[1:],
    )
    assert w1.sum() == approx(
        2 * weight * flux * p1.integral * c1.etendue / nfiles,
    )
    E = t1["PrimaryNeutrinoEnergy"]
    y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    assert y == approx(2 * weight * flux * Ewidth * c1.etendue / nfiles, 6e-3)


@pytest.mark.parametrize("weight", (0.1, 1, 10))
@pytest.mark.parametrize("nfiles", (1, 10, 100))
@pytest.mark.parametrize("flux", (1e-6, 1, 1e6))
def test_nugen_energy_pre_V04_00(weight, nfiles, flux):
    p1 = PowerLaw(0, 1e3, 1e4)
    c1 = CircleInjector(500, 0, 0.001)
    t1 = pd.DataFrame(make_new_table(14, 10000, c1, p1))
    t1["TotalInteractionProbabilityWeight"] = weight
    t1["InjectionSurfaceR"] = c1.radius
    f1 = {"I3MCWeightDict": t1}
    wf = NuGenWeighter(f1, nfiles=nfiles)
    w1 = wf.get_weights(flux)

    assert_allclose(
        w1[1:],
        flux
        * weight
        / (0.5 * t1["NEvents"] * nfiles * c1.pdf(np.cos(t1["PrimaryNeutrinoZenith"])) * p1.pdf(t1["PrimaryNeutrinoEnergy"]))[
            1:
        ],
    )
    assert w1.sum() == approx(
        2 * weight * flux * p1.integral * c1.etendue / nfiles,
    )
    E = t1["PrimaryNeutrinoEnergy"]
    y, x = np.histogram(E, weights=w1, bins=51, range=[p1.a, p1.b])
    Ewidth = np.ediff1d(x)
    assert y == approx(2 * weight * flux * Ewidth * c1.etendue / nfiles, 6e-3)


def test_empty():
    with pytest.raises(RuntimeError):
        x = {"I3MCWeightDict": {"PrimaryNeutrinoType": []}}
        NuGenWeighter(x, nfiles=1)


@pytest.mark.parametrize("weight", (0.1, 1, 10))
@pytest.mark.parametrize("flux", (1e-6, 1, 1e6))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_i3file_pre_V04_00(weight, flux):
    p1 = PowerLaw(0, 1e3, 1e4)
    c1 = CircleInjector(500, 0, 1)
    t1 = make_new_table(14, 1, c1, p1)
    mcw = dataclasses.I3MapStringDouble({n: t1[n][0] for n in t1.dtype.names})
    mcw["TotalInteractionProbabilityWeight"] = weight
    mcw["InjectionSurfaceR"] = c1.radius
    f1 = icetray.I3Frame()
    f1["I3MCWeightDict"] = mcw
    wf = NuGenWeighter(f1, nfiles=1)
    w1 = wf.get_weights(flux)
    assert w1 == approx(
        flux * weight / (0.5 * c1.pdf(np.cos(mcw["PrimaryNeutrinoZenith"])) * p1.pdf(mcw["PrimaryNeutrinoEnergy"]))
    )


@pytest.mark.parametrize("weight", (0.1, 1, 10))
@pytest.mark.parametrize("typeweight", (0.3, 0.5, 0.7))
@pytest.mark.parametrize("flux", (1e-6, 1, 1e6))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_i3file_pre_V6(weight, flux, typeweight):
    p1 = PowerLaw(0, 1e3, 1e4)
    c1 = UniformSolidAngleCylinder(1900, 950, 0, 1)
    t1 = make_new_table(14, 1, c1, p1)
    mcw = dataclasses.I3MapStringDouble({n: t1[n][0] for n in t1.dtype.names})
    mcw["TotalWeight"] = weight
    mcw["InjectionSurfaceR"] = -1
    mcw["TypeWeight"] = typeweight
    f1 = icetray.I3Frame()
    f1["I3MCWeightDict"] = mcw
    wf = NuGenWeighter(f1, nfiles=1)
    w1 = wf.get_weights(flux)
    assert w1 == approx(
        flux * weight / (typeweight * c1.pdf(np.cos(mcw["PrimaryNeutrinoZenith"])) * p1.pdf(mcw["PrimaryNeutrinoEnergy"]))
    )


@pytest.mark.parametrize("weight", (0.1, 1, 10))
@pytest.mark.parametrize("typeweight", (0.3, 0.5, 0.7))
@pytest.mark.parametrize("flux", (1e-6, 1, 1e6))
@pytest.mark.skipif("icetray" not in globals(), reason="Not in an IceTray environment")
def test_i3file_post_V6(weight, flux, typeweight):
    p1 = PowerLaw(0, 1e3, 1e4)
    c1 = UniformSolidAngleCylinder(200, 100, 0, 0.1)
    t1 = make_new_table(14, 1, c1, p1)
    mcw = dataclasses.I3MapStringDouble({n: t1[n][0] for n in t1.dtype.names})
    mcw["CylinderHeight"] = c1.length
    mcw["CylinderRadius"] = c1.radius
    mcw["TypeWeight"] = typeweight
    mcw["TotalWeight"] = weight
    f1 = icetray.I3Frame()
    f1["I3MCWeightDict"] = mcw
    wf = NuGenWeighter(f1, nfiles=1)
    w1 = wf.get_weights(flux)
    assert w1 == approx(
        flux * weight / (typeweight * c1.pdf(np.cos(mcw["PrimaryNeutrinoZenith"])) * p1.pdf(mcw["PrimaryNeutrinoEnergy"]))
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__, *sys.argv[1:]]))
