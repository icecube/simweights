#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import os.path
import sys
import tarfile
import tempfile
from pathlib import Path

from I3Tray import I3Tray
from icecube import dataclasses, hdfwriter, icetray, rootwriter, simclasses, tableio  # noqa: F401


def fake_event_header(frame: dict) -> None:
    header = dataclasses.I3EventHeader()
    header.run_id = 0
    header.event_id = fake_event_header.event_id
    fake_event_header.event_id += 1
    frame["I3EventHeader"] = header


fake_event_header.event_id = 0


filelist = [
    "/data/sim/IceCube/2015/filtered/level2/CORSIKA-in-ice/12602/0000000-0000999/Level2_IC86.2015_corsika.012602.000000.i3.bz2",
    "/data/sim/IceCube/2015/filtered/level2/CORSIKA-in-ice/20014/0000000-0000999/Level2_IC86.2015_corsika.020014.000000.i3.bz2",
    "/data/sim/IceCube/2015/filtered/level2/CORSIKA-in-ice/20021/0000000-0000999/Level2_IC86.2015_corsika.020021.000000.i3.bz2",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20208/0000000-0000999/Level2_IC86.2016_corsika.020208.000001.i3.bz2",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20243/0000000-0000999/Level2_IC86.2016_corsika.020243.000001.i3.bz2",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20263/0000000-0000999/Level2_IC86.2016_corsika.020263.000000.i3.zst",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20777/0000000-0000999/Level2_IC86.2016_corsika.020777.000000.i3.zst",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20778/0000000-0000999/Level2_IC86.2016_corsika.020778.000000.i3.zst",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20780/0000000-0000999/Level2_IC86.2016_corsika.020780.000000.i3.zst",
    "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/21889/0000000-0000999/Level2_IC86.2016_corsika.021889.000000.i3.zst",
    "/data/sim/IceCube/2011/filtered/level2/neutrino-generator/10634/00000-00999/Level2_IC86.2011_nugen_NuMu.010634.000000.i3.bz2",
    "/data/sim/IceCube/2011/filtered/level2/neutrino-generator/10692/00000-00999/Level2_IC86.2011_nugen_NuE.010692.000000.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11029/00000-00999/Level2_nugen_numu_IC86.2012.011029.000000.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11065/00000-00999/Level2_IC86.2012_nugen_NuTau.011065.000001.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/00000-00999/Level2_nugen_numu_IC86.2012.011069.000000.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11070/00000-00999/Level2_nugen_numu_IC86.2012.011070.000000.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11297/00000-00999/Level2_nugen_nutau_IC86.2012.011297.000000.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11477/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_nutau.011477.000000.clsim-base-4.0.3.0.99_eff.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11836/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_nutau.011836.000000.clsim-base-4.0.3.0.99_eff.i3.bz2",
    "/data/sim/IceCube/2012/filtered/level2/neutrino-generator/12646/0000000-0000999/clsim-base-4.0.5.0.99_eff/Level2_IC86.2012_nugen_nue.012646.000000.clsim-base-4.0.5.0.99_eff.i3.bz2",
    "/data/sim/IceCube/2016/filtered/level2/neutrino-generator/20878/0000000-0000999/Level2_IC86.2016_NuMu.020878.000000.i3.zst",
    "/data/sim/IceCube/2016/filtered/level2/neutrino-generator/20885/0000000-0000999/Level2_IC86.2016_NuE.020885.000000.i3.zst",
    "/data/sim/IceCube/2016/filtered/level2/neutrino-generator/20895/0000000-0000999/Level2_IC86.2016_NuTau.020895.000000.i3.zst",
    "/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012/SIBYLL2.1/p/12360_v1s/Level3_IC86.2012_SIBYLL2.1_p_12360_E6.0_0.i3.bz2",
    "/data/sim/IceCubeUpgrade/genie/step3/140021/upgrade_genie_step3_140021_000000.i3.zst",
]

if "notemp" in sys.argv:
    outdir = "/scratch/kmeagher/simweights/"
else:
    tempdir = tempfile.TemporaryDirectory(prefix="simweights_testdata_")
    outdir = Path(tempdir.name)

for filename in filelist:
    basename = Path(filename).name.replace(".i3.zst", "").replace(".i3.bz2", "").replace(".i3.gz", "")
    assert basename != Path(filename).name

    split = False
    if "corsika" in basename:
        keys = [
            "PolyplopiaPrimary",
            "I3PrimaryInjectorInfo",
            "I3CorsikaInfo",
            "CorsikaWeightMap",
            "I3CorsikaWeight",
        ]
        streams = ["InIceSplit"]
    elif "genie" in basename:
        keys = ["I3GenieInfo", "I3GenieResult", "I3MCWeightDict"]
        streams = ["NullSplit"]
        split = True
    elif "nugen" in basename or "Nu" in basename:
        keys = ["I3MCWeightDict"]
        streams = ["InIceSplit", "in_ice"]
    else:
        keys = ["I3TopInjectorInfo", "MCPrimary"]
        streams = ["IceTopSplit"]

    outfile = outdir / basename

    if Path(outfile.name + ".hdf5").exists():
        print(f"Skipping {filename}: {outfile} already exists!")
        continue

    print(f"Booking  : {filename}")
    print(f"  outfile: {outfile}")
    print(f"  keys   : {keys}")
    print(f"  streams: {streams}")

    tray = I3Tray()
    tray.Add("I3Reader", FileNameList=[filename])

    if split:
        tray.Add(
            fake_event_header,
            Streams=[icetray.I3Frame.DAQ],
            If=lambda f: "I3EventHeader" not in f,
        )
        tray.Add("I3NullSplitter", SubEventStreamName="NullSplit")
    tray.Add(
        tableio.I3TableWriter,
        tableservice=[
            hdfwriter.I3HDFTableService(str(outfile) + ".hdf5"),
            rootwriter.I3ROOTTableService(str(outfile) + ".root"),
        ],
        SubEventStreams=streams,
        keys=keys,
    )

    tray.Execute()
    del tray


tarfilename = "/data/user/kmeagher/simweights_testdata_test.tar.gz"
print(f"Writing tarfile {tarfilename}")

with tarfile.open(tarfilename, "w:gz") as tar:
    for f in os.listdir(outdir):
        print(f"Adding {f} to tarball")
        tar.add(outdir / f, arcname=f)

print("Done!")
