import argparse

import pandas as pd

import simweights
from simweights.utils import Null

default_flux_models = ["GaisserH3a"]

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+")
parser.add_argument("-f", "--flux", default=None)
args = parser.parse_args()

wobj = Null()


files = []

for fname in [
    "/home/kjm/icecube/corsika/dynstack/21668/0000000-0000999/IC86.2016_corsika.021668.N000009.hdf5"
]:
    f = pd.HDFStore(fname, "r")
    files.append(f)
    wobj += simweights.TriggeredCorsikaWeighter(f)

pdgids = set(wobj.surface.spectra.keys())

if args.flux is None:
    for f in default_flux_models:
        if pdgids == set(getattr(simweights, f).pdgids):
            model_name = f
            break
else:
    model_name = args.flux

flux_model = getattr(simweights, model_name)()

print(wobj.surface)
w = wobj.get_weights(flux_model)
print("Using flux model   : {}".format(model_name))
print("Number of Events   : {:8d}".format(w.size))
print("Event Rate         : {:8.6g} Hz".format(w.sum()))
print("Effective Area     : {:8.6g} m²".format(wobj.effective_area()[0][0]))
print("Effective Livetime : {:8.6g} s".format(w.sum() / (w ** 2).sum()))


[f.close() for f in files]
