#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pylab as plt
import simweights
import tables

# load hdf5 table
f = tables.open_file("Level2_IC86.2016_corsika.021682.N100.hdf5", "r")
wobj = simweights.CorsikaWeighter(f)
flux_model = simweights.GaisserH3a()
w = wobj.get_weights(flux_model)

# Select just the MuonFilter
w *= f.root.FilterMask.cols.MuonFilter_13[:][:, 1]

# print the total event rate and livetime
print(f"Event Rate    : {w.sum():6.2f} Hz")
print(f"Total Livetime: {w.sum() / (w**2).sum():6.2f} s")

# make bin edges from the energy range of the sample
Ebins = np.geomspace(*wobj.surface.get_energy_range(None), 50)

# get the energy column from the weight object
mcenergy = wobj.get_weight_column("energy")

# make histograms of the rate and the rate squared
h1, x2 = np.histogram(mcenergy, bins=Ebins, weights=w)
h2, x1 = np.histogram(mcenergy, bins=Ebins, weights=w**2)

# plot the rate
plt.step(np.r_[Ebins, Ebins[-1]], np.r_[0, h1, 0])
plt.semilogx()
plt.xlabel("Energy [GeV]")
plt.ylabel("Rate [Hz]")
plt.savefig("livetime_rate.svg")

# plot the livetime
plt.figure()
plt.step(np.r_[Ebins, Ebins[-1]], np.r_[0, h1 / h2, 0])
plt.semilogx()
plt.xlabel("Energy [GeV]")
plt.ylabel("Livetime [s]")
plt.savefig("livetime_livetime.svg")

plt.show()
