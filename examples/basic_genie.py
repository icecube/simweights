#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import nuflux
import pandas as pd
import simweights

simfile = pd.HDFStore("genie_reader_NuE_C_corr.hdf5")
flux_model = nuflux.makeFlux("IPhonda2014_spl_solmax")
weight_obj = simweights.GenieWeighter(simfile)
weights = weight_obj.get_weights(flux_model)
print("Rate", weights.sum(), "Hz")
