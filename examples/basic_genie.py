#!/usr/bin/env python3

import nuflux
import pandas

import simweights

simfile = pandas.HDFStore("genie_reader_NuE_C_corr.hdf5")
flux_model = nuflux.makeFlux("IPhonda2014_spl_solmax")
weight_obj = simweights.GenieWeighter(simfile)
weights = weight_obj.get_weights(flux_model)
print("Rate", weights.sum(), "Hz")
