#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import pandas as pd
import pylab as plt
import simweights
from numpy.typing import ArrayLike

# load the hdf5 file that we just created using pandas
hdffile = pd.HDFStore("Level2_IC86.2016_NuMu.021217.hdf5", "r")

# instantiate the weighter object by passing the pandas file to it
weighter = simweights.NuGenWeighter(hdffile, nfiles=10)


def northern_track(energy: ArrayLike) -> ArrayLike:
    """This function to represent the IceCube northern track limit.
    Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type.
    """
    return 1.44e-18 / 2 * (energy / 1e5) ** -2.2


# get the weights by passing the flux to the weighter
weights = weights = weighter.get_weights(northern_track)

# print some info about the weighting object
print(weighter.tostring(northern_track))

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
plt.tight_layout()
plt.show()
