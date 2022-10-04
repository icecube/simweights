#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from glob import glob

from I3Tray import I3Tray
from icecube import hdfwriter, simclasses

FILE_DIR = "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/21889/0000000-0000999"
file = sorted(glob(FILE_DIR + "/Level2_IC86.2016_corsika.021889.00000*.i3.zst"))

tray = I3Tray()
tray.Add("I3Reader", FileNameList=file)
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit"],
    keys=["PolyplopiaPrimary", "I3PrimaryInjectorInfo", "I3CorsikaWeight"],
    output="Level2_IC86.2016_corsika.021889.hdf5",
)

tray.Execute()
