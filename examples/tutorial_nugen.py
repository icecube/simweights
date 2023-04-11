#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import glob

import matplotlib.pyplot as plt
import nuflux
import numpy as np
import simweights
from icecube import dataclasses, dataio, simclasses


def get_most_energetic_muon(mmclist: simclasses.I3MMCTrackList) -> float:
    "Loop through the MMC track list and return the muon with the most energy."
    emax = 0
    for muon in list(mmclist):
        particle = muon.particle
        if (
            particle.type in (dataclasses.I3Particle.MuMinus, dataclasses.I3Particle.MuPlus)
            and particle.total_energy > emax
        ):
            emax = particle.total_energy
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

DATASET_DIR = "/data/sim/IceCube/2016/filtered/level2/neutrino-generator/21217/"
filelist = list(glob.glob(DATASET_DIR + "0000000-0000999/Level2_IC86.2016_NuMu.021217.00000*.i3.zst"))
MCmuonEnergy_nugen = np.array([])
I3MCWeightDict: dict = {k: [] for k in weight_keys}

for f in filelist:
    print("Reading", f)
    infile_nugen = dataio.I3File(f)
    while infile_nugen.more():
        frame = infile_nugen.pop_physics()
        if "FilterMask" in frame and frame["FilterMask"]["MuonFilter_13"].condition_passed:
            MCmuonEnergy_nugen = np.append(
                MCmuonEnergy_nugen,
                get_most_energetic_muon(frame["MMCTrackList"]),
            )
            for k in weight_keys:
                I3MCWeightDict[k].append(frame["I3MCWeightDict"][k])

nfiles = len(filelist)
wobj = simweights.NuGenWeighter({"I3MCWeightDict": I3MCWeightDict}, nfiles=nfiles)

# check that what we got matches what is in OneWeight
np.testing.assert_allclose(
    wobj.get_weights(1),
    np.array(I3MCWeightDict["OneWeight"]) / (0.5 * I3MCWeightDict["NEvents"][0] * nfiles),
)

conventional = nuflux.makeFlux("honda2006")
conventional.knee_reweighting_model = "gaisserH3a_elbert"
weights_simweights = wobj.get_weights(conventional)

erange = wobj.surface.get_energy_range(None)
czrange = wobj.surface.get_cos_zenith_range(None)
print(wobj.surface)
print(f"Number of files  : {nfiles:8d}")
print(f"Number of Events : {weights_simweights.size:8d}")
print(f"Effective Area   : {wobj.effective_area(erange, czrange)[0][0]:8.6g} m²")
print(f"Event Rate       : {weights_simweights.sum():8.6g} Hz")
print(f"Livetime         : {weights_simweights.sum() / (weights_simweights**2).sum():8.6g} s")

fig, ax = plt.subplots(figsize=(12, 6))
bins = np.logspace(1, 6, 25)
MCnuEnergy = wobj.get_weight_column("energy")
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
