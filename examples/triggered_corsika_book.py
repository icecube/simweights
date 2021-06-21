from I3Tray import I3Tray
from icecube import hdfwriter, simclasses

filedir = "/data/sim/scratch/IceSim/dynstack/21682/0000000-0000999/"
file = filedir + "Level2_IC86.2016_corsika.021682.000000.i3.zst"

tray = I3Tray()
tray.Add("I3Reader", FileNameList=[file])
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit"],
    keys=["PolyplopiaPrimary", "I3PrimaryInjectorInfo", "I3CorsikaWeight"],
    output="Level2_IC86.2016_corsika.021682.hdf5",
)

tray.Execute()
