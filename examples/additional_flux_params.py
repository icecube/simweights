import pandas as pd
import pylab as plt

# start-box1
import simweights

# load the hdf5 file and make the weigher
hdffile = pd.HDFStore("Level2_IC86.2016_NuMu.021217.N100.hdf5", "r")
weighter = simweights.NuGenWeighter(hdffile, nfiles=100)

# add a non-standard weighting column
weighter.add_weight_column("azimuth", weighter.get_column("PolyplopiaPrimary", "azimuth"))

# create a python function to that only depends on energy
# Note that the units are GeV^-1 * cm^-2 * sr^-1 * s^-1 per particle type
def simple_model(energy):
    return 1e-8 * energy**-2


# create another function that takes azimuth as a parameter. get_weights() will use the name
# of the function parameter to know which weighting column to access
def azimuthal_model(energy, azimuth):
    return 1e-8 * plt.cos(azimuth) ** 2 * energy**-2


for flux_function in (simple_model, azimuthal_model):
    # get the weights by passing the flux function to the weighter
    weights = weighter.get_weights(flux_function)

    # We can access our recently created weight column with get_weight_column()
    azimuth = weighter.get_weight_column("azimuth")

    # histogram the primary energy with the weights
    plt.hist(azimuth, weights=weights, bins=50, histtype="step", label=flux_function.__name__)
# end-box1

# make the plot look good
plt.legend(loc="best")
plt.xlabel("Azimuth [radian]")
plt.ylabel("Event Rate [Hz]")
plt.tight_layout()
plt.savefig("additional_flux_params.svg")
plt.show()
