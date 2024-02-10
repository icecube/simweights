#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
import os
import sys

import pytest
import tables

from simweights import CorsikaWeighter, GenieWeighter, IceTopWeighter, NuGenWeighter

with contextlib.suppress(ImportError):
    from icecube import dataio, simclasses  # noqa: F401

datadir = os.environ.get("SIMWEIGHTS_TESTDATA", None)

i3files = [
    ("Level3_IC86.2012_SIBYLL2.1_p_12360_E6.0_0", IceTopWeighter, None, 1),
    ("upgrade_genie_step3_141828_000000", GenieWeighter, None, 1),
    ("GENIE_NuMu_IceCubeUpgrade_v58.22590.000000", GenieWeighter, None, 1),
    ("Level2_IC86.2011_nugen_NuE.010692.000000", NuGenWeighter, 1, 1),
    ("Level2_IC86.2011_nugen_NuMu.010634.000000", NuGenWeighter, 1, 1),
    ("Level2_IC86.2012_nugen_nue.012646.000000.clsim-base-4.0.5.0.99_eff", NuGenWeighter, 1, 1),
    ("Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff", NuGenWeighter, 1, 1),
    ("Level2_IC86.2012_nugen_NuTau.011065.000001", NuGenWeighter, 1, 1),
    ("Level2_IC86.2012_nugen_nutau.011477.000000.clsim-base-4.0.3.0.99_eff", NuGenWeighter, 1, 1),
    ("Level2_IC86.2012_nugen_nutau.011836.000000.clsim-base-4.0.3.0.99_eff", NuGenWeighter, 1, 1),
    ("Level2_IC86.2016_NuE.020885.000000", NuGenWeighter, 1, 1),
    ("Level2_IC86.2016_NuMu.020878.000000", NuGenWeighter, 1, 1),
    ("Level2_IC86.2016_NuTau.020895.000000", NuGenWeighter, 1, 1),
    ("Level2_nugen_numu_IC86.2012.011029.000000", NuGenWeighter, 1, 1),
    ("Level2_nugen_numu_IC86.2012.011069.000000", NuGenWeighter, 1, 1),
    ("Level2_nugen_numu_IC86.2012.011070.000000", NuGenWeighter, 1, 1),
    ("Level2_nugen_nutau_IC86.2012.011297.000000", NuGenWeighter, 1, 1),
    ("Level2_IC86.2015_corsika.012602.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2015_corsika.020014.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2015_corsika.020021.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.020208.000001", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.020243.000001", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.020263.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.020777.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.020778.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.020780.000000", CorsikaWeighter, 1, 1),
    ("Level2_IC86.2016_corsika.021889.000000", CorsikaWeighter, None, 8),
]


@pytest.mark.parametrize(("fname", "weighter", "nfiles", "factor"), i3files)
@pytest.mark.skipif(not datadir, reason="environment variable SIMWEIGHTS_TESTDATA not set")
@pytest.mark.skipif("dataio" not in globals(), reason="Not in IceTray environment")
def test_i3_file(fname, weighter, nfiles, factor):
    args = {"nfiles": nfiles} if nfiles else {}
    with tables.open_file(datadir + "/" + fname + ".hdf5", "r") as hdf:
        hdfwght = weighter(hdf, **args)
        hdfw = hdfwght.get_weights(1)
    i3file = dataio.I3File(datadir + "/" + fname + ".i3.zst")
    i3w = []
    for frame in i3file:
        if frame.Stop != frame.DAQ:
            continue
        i3w.append(weighter(frame, **args).get_weights(1).item() / factor)
    assert i3w == pytest.approx(hdfw)


if __name__ == "__main__":
    pytest.main(["-v", __file__] + sys.argv[1:])
