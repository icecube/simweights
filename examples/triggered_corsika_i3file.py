#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from collections import defaultdict
from pathlib import Path

import numpy as np
from icecube import dataclasses, hdfwriter, icetray, simclasses

from simweights import CorsikaWeighter, GaisserH3a

FILE_DIR = Path("/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/21889/0000000-0000999")
files = sorted(str(f) for f in FILE_DIR.glob("Level2_IC86.2016_corsika.021889.000000.i3.zst"))


class Weighter(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.weighter = None
        self.fluxmodel = GaisserH3a()
        self.s_frames = defaultdict(int)

    def Simulation(self, frame):
        pdgid = frame["I3PrimaryInjectorInfo"].primary_type
        self.s_frames[pdgid] += 1

    def DAQ(self, frame):
        weighter = CorsikaWeighter(frame)
        weight = weighter.get_weights(self.fluxmodel)[0]
        name = f"weight_{self.fluxmodel.__class__.__name__}"
        frame[name] = dataclasses.I3Double(weight)
        self.PushFrame(frame)

    def Finish(self):
        print("Weights Need to be adjusted by the following factors:")
        for pdgid, counts in self.s_frames.items():
            print(f"{pdgid!s:>11} {counts}")


tray = icetray.I3Tray()
tray.Add("I3Reader", FileNameList=files)
tray.Add(Weighter)
tray.Execute()
