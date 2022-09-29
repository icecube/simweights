#!/usr/bin/env python3
import os.path
import tarfile
import tempfile

from I3Tray import I3Tray
from icecube import dataclasses, hdfwriter, icetray, rootwriter, simclasses, tableio


def fake_event_header(frame):
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
    "/data/user/kmeagher/genie_reader_NuE.i3.gz",
]

# tempdir = tempfile.TemporaryDirectory(prefix="simweights_testdata_")
outdir = "/scratch/kmeagher/simweights/"

for filename in filelist:
    basename = (
        os.path.basename(filename).replace(".i3.zst", "").replace(".i3.bz2", "").replace(".i3.gz", "")
    )
    assert basename != os.path.basename(filename)
    outfile = os.path.join(outdir, basename)

    split = False
    if "corsika" in basename:
        keys = [
            "PolyplopiaPrimary",
            "I3PrimaryInjectorInfo",
            "I3CorsikaInfo",
            "CorsikaWeightMap",
            "I3CorsikaWeight",
        ]
    elif "genie" in basename:
        keys = ["I3GenieInfo", "I3GenieResult", "I3MCWeightDict"]
        split = True
    else:
        keys = ["I3MCWeightDict"]

    print(f"Booking {filename} -> {outfile}")
    if os.path.exists(outfile + ".hdf5"):
        continue

    tray = I3Tray()
    tray.Add("I3Reader", FileNameList=[filename])

    if split:
        tray.Add(
            fake_event_header,
            Streams=[icetray.I3Frame.DAQ],
            If=lambda f: "I3EventHeader" not in f,
        )
        tray.Add("I3NullSplitter", SubEventStreamName="InIceSplit")
    tray.Add("Dump")
    tray.Add(
        tableio.I3TableWriter,
        tableservice=[
            hdfwriter.I3HDFTableService(outfile + ".hdf5"),
            rootwriter.I3ROOTTableService(outfile + ".root"),
        ],
        SubEventStreams=["InIceSplit", "in_ice"],
        keys=keys,
    )

    tray.Execute()
    del tray


tarfilename = "/data/user/kmeagher/simweights_testdata.tar.gz"
print(f"Writing tarfile {tarfilename}")

with tarfile.open(tarfilename, "w:gz") as tar:
    for f in os.listdir(outdir):
        print(f"Adding {f} to tarball")
        tar.add(os.path.join(outdir, f), arcname=f)

print("Done!")
