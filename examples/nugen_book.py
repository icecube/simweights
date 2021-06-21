from glob import glob

from I3Tray import I3Tray
from icecube import hdfwriter, simclasses

filedir = "/data/sim/IceCube/2016/filtered/level2/neutrino-generator/21217/0000000-0000999/"
files = sorted(glob(filedir + "Level2_IC86.2016_NuMu.021217.00000*.i3.zst"))

tray = I3Tray()
tray.Add("I3Reader", FileNameList=files)
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit"],
    keys=["PolyplopiaPrimary", "I3MCWeightDict"],
    output="Level2_IC86.2016_NuMu.021217.hdf5",
)

tray.Execute()
