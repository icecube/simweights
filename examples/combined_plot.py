#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pylab as plt
import simweights
import tables

# start-box1
# load the Medium Energy file
ME_file = tables.File("Level2_IC86.2016_corsika.021746.hdf5", "r")
ME_weighter = simweights.CorsikaWeighter(ME_file)

# load the High Energy file
HE_file = tables.File("Level2_IC86.2016_corsika.021745.hdf5", "r")
HE_weighter = simweights.CorsikaWeighter(HE_file)

# A combined weighter is created by summing two weighters
combined_weighter = ME_weighter + HE_weighter

# create a flux object and calculate the weights for all three weighters
flux_model = simweights.GaisserH3a()
ME_weights = ME_weighter.get_weights(flux_model)
HE_weights = HE_weighter.get_weights(flux_model)
combined_weights = combined_weighter.get_weights(flux_model)

# use Weighter.get_weight_column() to get the MC truth energy for each sample
ME_energy = ME_weighter.get_weight_column("energy")
HE_energy = HE_weighter.get_weight_column("energy")
combined_energy = combined_weighter.get_weight_column("energy")

# Histogram all three samples
Ebins = np.geomspace(3e4, 1e10, 64)
plt.hist(ME_energy, bins=Ebins, weights=ME_weights, histtype="step", label="Medium Energy")
plt.hist(HE_energy, bins=Ebins, weights=HE_weights, histtype="step", label="High Energy")
plt.hist(
    combined_energy,
    bins=Ebins,
    weights=combined_weights,
    histtype="step",
    color="k",
    ls="--",
    label="Combined",
)
# end-box1

plt.ylabel("Rate [Hz]")
plt.xlabel("MCTruth Energy [GeV]")
plt.legend(loc="best")
plt.loglog()
plt.tight_layout()
plt.savefig("combined_mctruth.svg")
plt.figure()

# start-box2
# use get_column() to return the Qtot for each sample
HE_Qtot = HE_weighter.get_column("Homogenized_QTot", "value")
ME_Qtot = ME_weighter.get_column("Homogenized_QTot", "value")
Combined_Qtot = combined_weighter.get_column("Homogenized_QTot", "value")

# histogram the Qtot
Qbins = np.geomspace(10, 1e6, 64)
plt.hist(ME_Qtot, bins=Qbins, weights=ME_weights, histtype="step", label="Medium Energy")
plt.hist(HE_Qtot, bins=Qbins, weights=HE_weights, histtype="step", label="High Energy")
plt.hist(
    Combined_Qtot,
    bins=Qbins,
    weights=combined_weights,
    histtype="step",
    label="Combined",
    ls="--",
    color="k",
)
# end-box2

plt.ylabel("Rate [Hz]")
plt.xlabel("Homogenized_QTot")
plt.legend(loc="best")
plt.loglog()
plt.tight_layout()
plt.savefig("combined_qtot.svg")
plt.show()
