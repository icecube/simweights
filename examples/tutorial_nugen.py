import glob

import matplotlib.pyplot as plt
import nuflux
import numpy as np
from icecube import dataclasses, dataio, simclasses
from icecube.icetray import I3Units

import simweights


def get_most_energetic_muon(mmclist):
    emax = 0
    for m in list(mmclist):
        p = m.particle
        if (
            p.type in (dataclasses.I3Particle.MuMinus, dataclasses.I3Particle.MuPlus)
            and p.total_energy > emax
        ):
            emax = p.total_energy
    return emax


weight_keys = [
    "MinZenith",
    "MaxZenith",
    "CylinderHeight",
    "CylinderRadius",
    "MinEnergyLog",
    "MaxEnergyLog",
    "PowerLawIndex",
    "PrimaryNeutrinoEnergy",
    "PrimaryNeutrinoType",
    "PrimaryNeutrinoZenith",
    "NEvents",
    "TotalWeight",
    "OneWeight",
]

dataset_dir = "/data/sim/IceCube/2016/filtered/level2/neutrino-generator/21217/"
filelist = list(glob.glob(dataset_dir + "0000000-0000999/Level2_IC86.2016_NuMu.021217.00000*.i3.zst"))
MCmuonEnergy_nugen = np.array([])
I3MCWeightDict = {k: [] for k in weight_keys}

for f in filelist:
    print("Reading", f)
    inFile_nugen = dataio.I3File(f)
    while inFile_nugen.more():
        frame = inFile_nugen.pop_physics()
        if "FilterMask" in frame:
            if frame["FilterMask"]["MuonFilter_13"].condition_passed:
                MCmuonEnergy_nugen = np.append(
                    MCmuonEnergy_nugen, get_most_energetic_muon(frame["MMCTrackList"])
                )
                for k in weight_keys:
                    I3MCWeightDict[k].append(frame["I3MCWeightDict"][k])

nfiles = len(filelist)
wobj = simweights.NuGenWeighter({"I3MCWeightDict": I3MCWeightDict}, nfiles=nfiles)

# check that what we got matches what is in OneWeight
units = I3Units.cm2 / I3Units.m2
np.testing.assert_allclose(
    wobj.get_weights(1),
    np.array(I3MCWeightDict["OneWeight"]) / (0.5 * I3MCWeightDict["NEvents"][0] * nfiles) * units,
)

conventional = nuflux.makeFlux("honda2006")
conventional.knee_reweighting_model = "gaisserH3a_elbert"
weights_simweights = wobj.get_weights(conventional)
print(wobj.surface)
print("Number of files  : {:8d}".format(nfiles))
print("Number of Events : {:8d}".format(weights_simweights.size))
print("Effective Area   : {:8.6g} m²".format(wobj.effective_area()[0][0]))
print("Event Rate       : {:8.6g} Hz".format(weights_simweights.sum()))
print("Livetime         : {:8.6g} s".format(weights_simweights.sum() / (weights_simweights ** 2).sum()))

fig, ax = plt.subplots(figsize=(12, 6))
bins = np.logspace(1, 6, 25)
MCnuEnergy = wobj.get_column("I3MCWeightDict", "PrimaryNeutrinoEnergy")
plt.hist(MCnuEnergy, bins=bins, histtype="step", weights=weights_simweights, label="Primary Energy")
plt.hist(MCmuonEnergy_nugen, bins=bins, histtype="step", weights=weights_simweights, label="Muon Energy")

plt.loglog()
plt.legend()
plt.xlabel("True Energy", fontsize=14)
plt.ylabel("events", fontsize=14)
plt.ylim(1e-6, 1e-2)
ax.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()