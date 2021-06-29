import pandas as pd
import pylab as plt

import simweights

# load the hdf5 file that we just created using pandas
hdffile = pd.HDFStore("Level2_IC86.2016_NuMu.021217.hdf5", "r")

# instantiate the weighter object by passing the pandas file to it
weighter = simweights.NuGenWeighter(hdffile, nfiles=10)

# create an function to represent the Waxman and Bahcall limit
# Note that the units are GeV^-1 * m^-2 * sr^-1 * s^-1 per particle type
def wblimit(energy):
    return 1e-4 * energy ** -2


# get the weights by passing the flux to the weighter
weights = weights = weighter.get_weights(wblimit)

# print some info about the weighting object
print(weighter.tostring(wblimit))

# create equal spaced bins in log space
bins = plt.geomspace(1e2, 1e8, 50)

# get energy of the primary cosmic-ray from `PolyplopiaPrimary`
primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")

# histogram the primary energy with the weights
plt.hist(primary_energy, weights=weights, bins=bins)

# make the plot look good
plt.loglog()
plt.xlabel("Primary Energy [GeV]")
plt.ylabel("Event Rate [Hz]")
plt.xlim(bins[0], bins[-1])
plt.ylim(1e-8, 2e-6)
plt.show()
