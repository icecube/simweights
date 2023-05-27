#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from glob import glob

import numpy as np
import pylab as plt
import simweights
import tables

# —- Define which files you want to be part of the dataset ——
# (This particular example is all low-energy protons)
FILE_DIR = "/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012/SIBYLL2.1/p/12360_v1s/h5files/"
bigfilelist = sorted(glob(FILE_DIR + "Level3_IC86.2012_SIBYLL2.1_p_12360_E*.h5"))

# —- Create the “IceTopWeighter” ——
# In this case, it's from a whole bunch of files strung together using the "+" operator.
# (This example uses tables, but one could also use pandas if you prefer)
weighter = None
for filename in bigfilelist:
    file_obj = tables.open_file(filename, "r")
    if weighter is None:
        weighter = simweights.IceTopWeighter(file_obj)
    else:
        weighter += simweights.IceTopWeighter(file_obj)

# -- Choose a flux model --
# This particular one is the "four-component" Gaisser H4a model (p, He, O, Fe).
# Note the "_IT" in the name of the function.  This distinguishes it from the five-component
# (p, He, N, Al, Fe) version of this model.
flux = simweights.GaisserH4a_IT()

# Get the weights by passing the flux to the weighter
weights = weighter.get_weights(flux)

# Dump some info to the screen about the weighting object, if you like
print(weighter.tostring(flux))

# —- Make a plot of your favorite variable ——
# ...such as primary energy.
primary_energy = weighter.get_column("MCPrimary", "energy")
plt.hist(
    np.log10(primary_energy),
    weights=weights,
    bins=46,
    range=[5.0, 9.6],
    log=True,
    histtype="step",
)
plt.xlabel("log10(Primary Energy [GeV])")
plt.ylabel("Event Rate [Hz]")
plt.show()
