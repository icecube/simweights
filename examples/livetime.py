import tables
import simweights
import pylab as plt
import numpy as np

# load hdf5 table
f = tables.open_file("Level2_IC86.2016_corsika.021682.N100.hdf5", "r")
wobj = simweights.CorsikaWeighter(f)
flux_model = simweights.GaisserH3a()
w = wobj.get_weights(flux_model)

# Select just the MuonFilter
w *= f.root.FilterMask.cols.MuonFilter_13[:][:, 1]

# print the total event rate and livetime
print("Event Rate    : {:6.2f} Hz".format(w.sum()))
print("Total Livetime: {:6.2f} s".format(w.sum() / (w ** 2).sum()))

# make bin edges from the energy range of the sample
Ebins = np.geomspace(*wobj.surface.get_energy_range(None), 50)

# get the energy column from the weight object
mcenergy = wobj.get_weight_column("energy")

# make histograms of the rate and the rate squared
h1, x2 = np.histogram(mcenergy, bins=Ebins, weights=w)
h2, x1 = np.histogram(mcenergy, bins=Ebins, weights=w ** 2)

# plot the rate
plt.step(Ebins, np.r_[0, h1])
plt.semilogx()
plt.xlabel("Energy [GeV]")
plt.ylabel("Rate [Hz]")

# plot the livetime
plt.figure()
plt.step(Ebins, np.r_[0, h1 / h2])
plt.semilogx()
plt.xlabel("Energy [GeV]")
plt.ylabel("Livetime [s]")

plt.show()