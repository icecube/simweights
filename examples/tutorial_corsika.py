import glob

import matplotlib.pyplot as plt
import numpy as np
from icecube import dataio

import simweights

corsika_dataset_dir = "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/20904/"
corsika_filelist = list(
    glob.glob(corsika_dataset_dir + "0000000-0000999/Level2_IC86.2016_corsika.020904.00000*.i3.zst")
)

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

CorsikaWeightMap = {k: [] for k in weight_keys}
PolyplopiaPrimary = {k: [] for k in ["type", "energy", "zenith"]}
MCtype_corsika = np.array([])
MCenergy_corsika = np.array([])

for f in corsika_filelist:
    print("Reading", f)
    inFile_corsika = dataio.I3File(f)
    while inFile_corsika.more():
        frame = inFile_corsika.pop_physics()
        if "FilterMask" in frame:
            if frame["FilterMask"]["MuonFilter_13"].condition_passed:
                # Frame may contain coincident events so select injected primary shower 'PolyplopiaPrimary'
                MCtype_corsika = np.append(MCtype_corsika, frame["PolyplopiaPrimary"].type)
                MCenergy_corsika = np.append(MCenergy_corsika, frame["PolyplopiaPrimary"].energy)

                for k in weight_keys:
                    CorsikaWeightMap[k].append(frame["CorsikaWeightMap"][k])

                PolyplopiaPrimary["zenith"].append(frame["PolyplopiaPrimary"].dir.zenith)
                PolyplopiaPrimary["type"].append(frame["PolyplopiaPrimary"].type)
                PolyplopiaPrimary["energy"].append(frame["PolyplopiaPrimary"].energy)

fobj = dict(CorsikaWeightMap=CorsikaWeightMap, PolyplopiaPrimary=PolyplopiaPrimary)
wobj = simweights.CorsikaWeighter(fobj, nfiles=len(corsika_filelist))
weights_GaisserH3a = wobj.get_weights(simweights.GaisserH3a())
weights_Hoerandel = wobj.get_weights(simweights.Hoerandel())

weightssqr_GaisserH3a = np.power(weights_GaisserH3a, 2)
livetime_GaisserH3a = weights_GaisserH3a / weightssqr_GaisserH3a

weightssqr_Hoerandel = np.power(weights_Hoerandel, 2)
livetime_Hoerandel = weights_GaisserH3a / weightssqr_Hoerandel

erange = wobj.surface.get_energy_range(None)
czrange = wobj.surface.get_cos_zenith_range(None)
print(f"Number of files  : {len(corsika_filelist)}")
print(f"Number of events : {len(weights_GaisserH3a)}")
print(f"Effective Area   : {wobj.effective_area(erange, czrange)[0][0]:10.2} m²")

print("           GaisserH3a Hoerandel")
print(f"Rate     : {weights_GaisserH3a.sum():10.4f} {weights_Hoerandel.sum():10.4f}")
print(
    "Livetime : {:10.6f} {:10.6f}".format(
        weights_GaisserH3a.sum() / (weightssqr_GaisserH3a.sum()),
        weights_Hoerandel.sum() / weightssqr_GaisserH3a.sum(),
    )
)

fig, ax = plt.subplots(figsize=(12, 6))
bins = np.logspace(3, 9, 50)
plt.hist(MCenergy_corsika, bins=bins, histtype="step", weights=weights_GaisserH3a, label="GaisserH3a")
plt.hist(MCenergy_corsika, bins=bins, histtype="step", weights=weights_Hoerandel, label="Hoerandel")

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

histweightsG, cpuedges = np.histogram(MCenergy_corsika, bins=bins, weights=weights_GaisserH3a)
histweights2G, cpuedges2 = np.histogram(MCenergy_corsika, bins=bins, weights=weightssqr_GaisserH3a)
plt.plot(bins[1:], histweightsG / histweights2G, label="Livetime (H3a)")

histweightsH, cpuedges = np.histogram(MCenergy_corsika, bins=bins, weights=weights_Hoerandel)
histweights2H, cpuedges2 = np.histogram(MCenergy_corsika, bins=bins, weights=weightssqr_Hoerandel)
plt.plot(bins[1:], histweightsH / histweights2H, label="Livetime (Hoerandel)")

plt.loglog()
plt.legend()
plt.xlabel("Primary Energy [GeV]", fontsize=14)
plt.ylabel("Livetime [s]", fontsize=14)
plt.ylim(1e1, 1e7)
ax.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()
