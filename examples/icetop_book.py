#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from glob import glob

from I3Tray import I3Tray
from icecube import hdfwriter, simclasses

# The following is an example of some "Level3" files for which S-frames were retrofitted
# after production.  But the input can be any IceTop simulation files containing S-frames.
FILE_DIR = "/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012/SIBYLL2.1/p/12360_v1s/"
files = sorted(glob(FILE_DIR + "Level3_IC86.2012_SIBYLL2.1_p_12360_E6.0_*.i3.bz2"))

tray = I3Tray()
tray.Add("I3Reader", FileNameList=files)
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["IceTopSplit"],
    keys=["I3TopInjectorInfo"],  # <--- and of course whatever other keys you want... MCPrimary, etc...
    output="Level3_IC86.2012_SIBYLL2.1_p_12360_E6.0.i3.bz2",
)

tray.Execute()
