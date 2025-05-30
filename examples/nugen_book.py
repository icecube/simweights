#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from pathlib import Path

from icecube import hdfwriter, icetray, simclasses

FILE_DIR = Path("/data/sim/IceCube/2016/filtered/level2/neutrino-generator/21217/0000000-0000999/")
files = sorted(str(f) for f in FILE_DIR.glob("Level2_IC86.2016_NuMu.021217.0000*.i3.zst"))
nfiles = len(files)

tray = icetray.I3Tray()
tray.Add("I3Reader", FileNameList=files)
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit"],
    keys=["PolyplopiaPrimary", "I3MCWeightDict", "FilterMask"],
    output=f"Level2_IC86.2016_NuMu.021217.N{nfiles}.hdf5",
)

tray.Execute()
