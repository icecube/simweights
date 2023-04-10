#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from glob import glob

import pylab as plt
import simweights
from icecube import dataio, simclasses  # noqa: F401

FILE_DIR = "/data/sim/IceCube/2016/filtered/level2/CORSIKA-in-ice/21889/0000000-0000999"
filelist = sorted(glob(FILE_DIR + "/Level2_IC86.2016_corsika.021889.00000*.i3.zst"))
assert filelist

# SimWeights iterates over the info table row-wise so to mimic it we need a list
I3PrimaryInjectorInfo = []

# However, the weight object is iterated column-wise so we need to create a dictionary of lists
I3CorsikaWeight: dict = {"energy": [], "type": [], "zenith": [], "weight": []}

# loop over all the files we want to read
for filename in filelist:
    # open the i3 files with the dataio interface
    infile = dataio.I3File(filename)
    print("Reading " + filename)

    # loop over the frames in the file
    while infile.more():
        # get the frame
        frame = infile.pop_frame()

        # if this is an S-Frame
        if frame.Stop == frame.Simulation:
            # get the info from the frame
            info = frame["I3PrimaryInjectorInfo"]

            # convert the info object into a dictionary which represents a row
            inforow = {x: getattr(info, x) for x in dir(info) if x[0] != "_"}

            # add the row to to the list
            I3PrimaryInjectorInfo.append(inforow)

        # if this is a physics event in the right sub-event stream
        elif frame.Stop == frame.Physics and frame["I3EventHeader"].sub_event_stream == "InIceSplit":
            # get the weighting object
            w = frame["I3CorsikaWeight"]

            # for each of the columns we need get it from the frame object
            # and put it in the correct column
            I3CorsikaWeight["energy"].append(w.primary.energy)
            I3CorsikaWeight["type"].append(w.primary.type)
            I3CorsikaWeight["zenith"].append(w.primary.dir.zenith)
            I3CorsikaWeight["weight"].append(w.weight)

# make a dictionary object to mimic the file structure of a pandas file
fileobj = {"I3PrimaryInjectorInfo": I3PrimaryInjectorInfo, "I3CorsikaWeight": I3CorsikaWeight}

# create the weighter object
weighter = simweights.CorsikaWeighter(fileobj)

# create an object to represent our cosmic-ray primary flux model
flux = simweights.GaisserH4a()

# get the weights by passing the flux to the weighter
weights = weights = weighter.get_weights(flux)

# print some info about the weighting object
print(weighter.tostring(flux))

# create equal spaced bins in log space
bins = plt.geomspace(3e4, 1e6, 50)

# get energy of the primary cosmic-ray from `PolyplopiaPrimary`
primary_energy = weighter.get_weight_column("energy")

# histogram the primary energy with the weights
plt.hist(primary_energy, weights=weights, bins=bins)

# make the plot look good
plt.xlabel("Primary Energy [GeV]")
plt.ylabel("Event Rate [Hz]")
plt.xlim(bins[0], bins[-1])
plt.ylim(0.1, 10)
plt.loglog()
plt.savefig("without_tableio.svg")
plt.show()
