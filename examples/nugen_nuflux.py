import pandas as pd
import pylab as plt
import nuflux

# start-box1
import simweights

# load the hdf5 file and make the weigher
hdffile = pd.HDFStore("Level2_IC86.2016_NuMu.021217.hdf5", "r")
weighter = simweights.NuGenWeighter(hdffile, nfiles=10)
bins = plt.geomspace(1e2, 1e8, 50)
primary_energy = weighter.get_column("PolyplopiaPrimary", "energy")

# create an function to represent the IceCube northern track limit
# Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type
def NorthernTrack(energy):
    return 1.44e-18 / 2 * (energy/1e5)**-2.2
NorthernTrack.name = "Northern Track 9.5 year"

# Create models and put them in a list so we can iterate over them
models = [
    NorthernTrack,
    nuflux.makeFlux("CORSIKA_GaisserH3a_average"),
    nuflux.makeFlux("H3a_SIBYLL23C"),
    nuflux.makeFlux("honda2006"),
    nuflux.makeFlux("BERSS_H3p_central"),
    nuflux.makeFlux("sarcevic_std"),
]

for flux_model in models:
    # get the weights by passing the flux to the weighter
    weights = weighter.get_weights(flux_model)

    # print the total rate of each model
    print("{:26} {:8.2f} mHz".format(flux_model.name, 1e6 * weights.sum()))

    # histogram the primary energy with the weights
    plt.hist(primary_energy, weights=weights, bins=bins, 
        histtype="step", label=flux_model.name)
# end-box1

# make the plot look good
plt.tight_layout()
plt.legend(loc=3)
plt.loglog()
plt.xlabel("Primary Energy [GeV]")
plt.ylabel("Event Rate [Hz]")
plt.xlim(bins[0], bins[-1])
plt.savefig("nugen_nuflux.svg")
plt.show()