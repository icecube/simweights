#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pylab as plt
import simweights
import tables

# start-example1
# load hdf5 file
f = tables.open_file("Level2_IC86.2016_NuMu.021217.N100.hdf5", "r")
# create weighter object
w = simweights.NuGenWeighter(f, nfiles=10)
# create the energy and zenith bins
energy_bins = np.geomspace(1e2, 1e8, 25)
zenith_bins = [-1, -0.5, 0, 0.5, 1]

# calculate the effective area in energy and zenith bins with all of the events in the sample
effective_area = w.effective_area(energy_bins, zenith_bins)

# make some labels for the different zenith bins
zenith_labels = [
    "120° < Zenith < 180°",
    "\u200790° < Zenith < 120°",
    "\u200760° < Zenith < \u200790°",
    "\u2007\u20070° < Zenith < \u200760°",
]

# for each zenith bin plot effective area as a function of energy
for i, zenith_slice in enumerate(effective_area):
    plt.step(energy_bins, np.r_[0, zenith_slice], label=zenith_labels[i])
# end-example1

plt.legend(loc="best")
plt.xlabel("Energy [GeV]")
plt.ylabel("Effective Area [m$^2$]")
plt.xlim(energy_bins[0], energy_bins[-1])
plt.loglog()
plt.tight_layout()
plt.savefig("effective_area_zenith.svg")

plt.figure()

# start-example2
# use a mask to select just the NuMus
numu_mask = w.get_weight_column("pdgid") == simweights.PDGCode.NuMu
ea_numu = w.effective_area(energy_bins, [-1, 1], numu_mask)
plt.step(energy_bins, np.r_[0, ea_numu[0]], label=r"$\nu_{\mu}$")

# Now use a mask to select just the NuMuBars
numubar_mask = w.get_weight_column("pdgid") == simweights.PDGCode.NuMuBar
ea_numubar = w.effective_area(energy_bins, [-1, 1], numubar_mask)
plt.step(energy_bins, np.r_[0, ea_numubar[0]], label=r"$\bar{\nu}_{\mu}$")

# not using any mask will calculate the average
ea_avg = w.effective_area(energy_bins, [-1, 1])
plt.step(energy_bins, np.r_[0, ea_avg[0]], label="Average")
# end-example2

plt.legend(loc="best")
plt.xlabel("Energy [GeV]")
plt.ylabel("Effective Area [m$^2$]")
plt.xlim(energy_bins[0], energy_bins[-1])
plt.loglog()
plt.tight_layout()
plt.savefig("effective_area_type.svg")

plt.figure()

# start-example3
# list of the filters we want to show
filters = ["MuonFilter_13", "GRECOOnlineFilter_19", "HESEFilter_15"]

# loop over the filters
for filter_name in filters:
    # create a mask
    mask = getattr(f.root.FilterMask.cols, filter_name)[:][:, 1].astype(bool)
    if np.any(mask):
        EA = w.effective_area(energy_bins, [-1, 1], mask)
        plt.step(energy_bins, np.r_[0, EA[0]], label=filter_name)
# end-example3

plt.legend(loc="best")
plt.xlabel("Energy [GeV]")
plt.ylabel("Effective Area [m$^2$]")
plt.xlim(energy_bins[0], energy_bins[-1])
plt.loglog()
plt.tight_layout()
plt.savefig("effective_area_filters.svg")

plt.show()
