#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import os
import sys

import pytest
import tables
from numpy.testing import assert_allclose
from simweights import CorsikaWeighter, GenieWeighter, IceTopWeighter, NuGenWeighter

try:
    from icecube import dataio, simclasses  # noqa: F401
except ImportError:
    dataio = None

datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)


def compare_i3_file(filename, weighter, factor=1, **kwargs):
    if dataio is None:
        pytest.skip("Not in IceTray environment")
    if datadir is None:
        pytest.skip("environment variable SIMWEIGHTS_TESTDATA not set")
    with tables.open_file(datadir + "/" + filename + ".hdf5", "r") as hdf:
        hdfwght = weighter(hdf, **kwargs)
        hdfw = hdfwght.get_weights(1)
    i3file = dataio.I3File(datadir + "/" + filename + ".i3.zst")
    i3w = []
    for frame in i3file:
        if frame.Stop != frame.DAQ:
            continue
        i3w.append(weighter(frame, **kwargs).get_weights(1).item() / factor)

    assert_allclose(i3w, hdfw)


def test_12602():
    compare_i3_file("Level2_IC86.2015_corsika.012602.000000", CorsikaWeighter, nfiles=1)


def test_20014():
    compare_i3_file("Level2_IC86.2015_corsika.020014.000000", CorsikaWeighter, nfiles=1)


def test_20021():
    compare_i3_file("Level2_IC86.2015_corsika.020021.000000", CorsikaWeighter, nfiles=1)


def test_20208():
    compare_i3_file("Level2_IC86.2016_corsika.020208.000001", CorsikaWeighter, nfiles=1)


def test_20243():
    compare_i3_file("Level2_IC86.2016_corsika.020243.000001", CorsikaWeighter, nfiles=1)


def test_20263():
    compare_i3_file("Level2_IC86.2016_corsika.020263.000000", CorsikaWeighter, nfiles=1)


def test_20777():
    compare_i3_file("Level2_IC86.2016_corsika.020777.000000", CorsikaWeighter, nfiles=1)


def test_20778():
    compare_i3_file("Level2_IC86.2016_corsika.020778.000000", CorsikaWeighter, nfiles=1)


def test_20780():
    compare_i3_file("Level2_IC86.2016_corsika.020780.000000", CorsikaWeighter, nfiles=1)


def test_21889():
    compare_i3_file("Level2_IC86.2016_corsika.021889.000000", CorsikaWeighter, factor=8)


def test_10634():
    compare_i3_file("Level2_IC86.2011_nugen_NuMu.010634.000000", NuGenWeighter, nfiles=1)


def test_10692():
    compare_i3_file("Level2_IC86.2011_nugen_NuE.010692.000000", NuGenWeighter, nfiles=1)


def test_11029():
    compare_i3_file("Level2_nugen_numu_IC86.2012.011029.000000", NuGenWeighter, nfiles=1)


def test_11065():
    compare_i3_file("Level2_IC86.2012_nugen_NuTau.011065.000001", NuGenWeighter, nfiles=1)


def test_11069():
    compare_i3_file("Level2_nugen_numu_IC86.2012.011069.000000", NuGenWeighter, nfiles=1)


def test_11070():
    compare_i3_file("Level2_nugen_numu_IC86.2012.011070.000000", NuGenWeighter, nfiles=1)


def test_11297():
    compare_i3_file("Level2_nugen_nutau_IC86.2012.011297.000000", NuGenWeighter, nfiles=1)


def test_11374():
    compare_i3_file("Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff", NuGenWeighter, nfiles=1)


def test_11477():
    compare_i3_file("Level2_IC86.2012_nugen_nutau.011477.000000.clsim-base-4.0.3.0.99_eff", NuGenWeighter, nfiles=1)


def test_11836():
    compare_i3_file("Level2_IC86.2012_nugen_nutau.011836.000000.clsim-base-4.0.3.0.99_eff", NuGenWeighter, nfiles=1)


def test_12646():
    compare_i3_file("Level2_IC86.2012_nugen_nue.012646.000000.clsim-base-4.0.5.0.99_eff", NuGenWeighter, nfiles=1)


def test_20878():
    compare_i3_file("Level2_IC86.2016_NuMu.020878.000000", NuGenWeighter, nfiles=1)


def test_20885():
    compare_i3_file("Level2_IC86.2016_NuE.020885.000000", NuGenWeighter, nfiles=1)


def test_20895():
    compare_i3_file("Level2_IC86.2016_NuTau.020895.000000", NuGenWeighter, nfiles=1)


def test_22590():
    compare_i3_file("GENIE_NuMu_IceCubeUpgrade_v58.22590.000000", GenieWeighter)


def test_141828():
    compare_i3_file("upgrade_genie_step3_141828_000000", GenieWeighter)


def test_icetop():
    compare_i3_file("Level3_IC86.2012_SIBYLL2.1_p_12360_E6.0_0", IceTopWeighter)


if __name__ == "__main__":
    pytest.main(["-v", __file__] + sys.argv[1:])
