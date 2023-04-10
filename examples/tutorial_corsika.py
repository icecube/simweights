#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import glob

import matplotlib.pyplot as plt
import numpy as np
import simweights
from icecube import dataio

CORSIKA_DATASET_DIR = "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20904/"
corsika_filelist = list(
    glob.glob(CORSIKA_DATASET_DIR + "0000000-0000999/Level2_IC86.2016_corsika.020904.00000*.i3.zst"),
)
assert corsika_filelist

weight_keys = [
    "CylinderLength",
    "CylinderRadius",
    "EnergyPrimaryMax",
    "EnergyPrimaryMin",
    "NEvents",
    "OverSampling",
    "ParticleType",
    "PrimaryEnergy",
    "PrimarySpectralIndex",
    "PrimaryType",
    "ThetaMax",
    "ThetaMin",
    "Weight",
]

particle_keys = ["type", "energy", "zenith"]

CorsikaWeightMap: dict = {k: [] for k in weight_keys}
PolyplopiaPrimary: dict = {k: [] for k in ["type", "energy", "zenith"]}
MCtype_corsika = np.array([])
MCenergy_corsika = np.array([])

for f in corsika_filelist:
    print("Reading", f)
    infile_corsika = dataio.I3File(f)
    while infile_corsika.more():
        frame = infile_corsika.pop_physics()
        if "FilterMask" in frame and frame["FilterMask"]["MuonFilter_13"].condition_passed:
            # Frame may contain coincident events so select injected primary shower 'PolyplopiaPrimary'
            MCtype_corsika = np.append(MCtype_corsika, frame["PolyplopiaPrimary"].type)
            MCenergy_corsika = np.append(MCenergy_corsika, frame["PolyplopiaPrimary"].energy)

            for k in weight_keys:
                CorsikaWeightMap[k].append(frame["CorsikaWeightMap"][k])

            PolyplopiaPrimary["zenith"].append(frame["PolyplopiaPrimary"].dir.zenith)
            PolyplopiaPrimary["type"].append(frame["PolyplopiaPrimary"].type)
            PolyplopiaPrimary["energy"].append(frame["PolyplopiaPrimary"].energy)

fobj = {"CorsikaWeightMap": CorsikaWeightMap, "PolyplopiaPrimary": PolyplopiaPrimary}
wobj = simweights.CorsikaWeighter(fobj, nfiles=len(corsika_filelist))
Weights_GaisserH3a = wobj.get_weights(simweights.GaisserH3a())
Weights_Hoerandel = wobj.get_weights(simweights.Hoerandel())

WeightsSqr_GaisserH3a = np.power(Weights_GaisserH3a, 2)
Livetime_GaisserH3a = Weights_GaisserH3a / WeightsSqr_GaisserH3a

WeightsSqr_Hoerandel = np.power(Weights_Hoerandel, 2)
Livetime_Hoerandel = Weights_GaisserH3a / WeightsSqr_Hoerandel

erange = wobj.surface.get_energy_range(None)
czrange = wobj.surface.get_cos_zenith_range(None)
print(f"Number of files  : {len(corsika_filelist)}")
print(f"Number of events : {len(Weights_GaisserH3a)}")
print(f"Effective Area   : {wobj.effective_area(erange, czrange)[0][0]:10.2} m²")

print("           GaisserH3a Hoerandel")
print(f"Rate     : {Weights_GaisserH3a.sum():10.4f} {Weights_Hoerandel.sum():10.4f}")
print(
    f"Livetime : {Weights_GaisserH3a.sum() / (WeightsSqr_GaisserH3a.sum()):10.6f} "
    f"{Weights_Hoerandel.sum() / WeightsSqr_GaisserH3a.sum():10.6f}",
)

fig, ax = plt.subplots(figsize=(12, 6))
bins = np.logspace(3, 9, 50)
plt.hist(MCenergy_corsika, bins=bins, histtype="step", weights=Weights_GaisserH3a, label="GaisserH3a")
plt.hist(MCenergy_corsika, bins=bins, histtype="step", weights=Weights_Hoerandel, label="Hoerandel")

plt.loglog()
plt.legend()
plt.xlabel("Primary Energy [GeV]", fontsize=14)
plt.ylabel("Event Rate [Hz]", fontsize=14)
plt.ylim(1e-6, 3)
ax.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig, ax = plt.subplots(figsize=(12, 6))
bins = np.logspace(3, 9, 50)

histweightsg, cpuedges = np.histogram(MCenergy_corsika, bins=bins, weights=Weights_GaisserH3a)
histweights2g, cpuedges2 = np.histogram(MCenergy_corsika, bins=bins, weights=WeightsSqr_GaisserH3a)
plt.plot(bins[1:], histweightsg / histweights2g, label="Livetime (H3a)")

histweightsh, cpuedges = np.histogram(MCenergy_corsika, bins=bins, weights=Weights_Hoerandel)
histweights2h, cpuedges2 = np.histogram(MCenergy_corsika, bins=bins, weights=WeightsSqr_Hoerandel)
plt.plot(bins[1:], histweightsh / histweights2h, label="Livetime (Hoerandel)")

plt.loglog()
plt.legend()
plt.xlabel("Primary Energy [GeV]", fontsize=14)
plt.ylabel("Livetime [s]", fontsize=14)
plt.ylim(1e1, 1e7)
ax.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()
