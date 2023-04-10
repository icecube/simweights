#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import pandas as pd
import pylab as plt
import simweights

# load the hdf5 file that we just created using pandas
hdffile = pd.HDFStore("Level2_IC86.2016_corsika.021889.hdf5", "r")

# instantiate the weighter object by passing the pandas file to it
weighter = simweights.CorsikaWeighter(hdffile)

# create an object to represent our cosmic-ray primary flux model
flux = simweights.GaisserH4a()

# get the weights by passing the flux to the weighter
weights = weights = weighter.get_weights(flux)

# print some info about the weighting object
print(weighter.tostring(flux))

# create equal spaced bins in log space
bins = plt.geomspace(3e4, 1e6, 50)

# get energy of the primary cosmic-ray from `PolyplopiaPrimary`
primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")

# histogram the primary energy with the weights
plt.hist(primary_energy, weights=weights, bins=bins)

# make the plot look good
plt.loglog()
plt.xlabel("Primary Energy [GeV]")
plt.ylabel("Event Rate [Hz]")
plt.xlim(bins[0], bins[-1])
plt.ylim(0.1, 10)
plt.savefig("triggered_corsika_tutorial.svg")
plt.show()
