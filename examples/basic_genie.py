#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import nuflux
import pandas as pd

import simweights

simfile = pd.HDFStore("upgrade_genie_step3_140021_000000.hdf5", "r")
flux_model = nuflux.makeFlux("IPhonda2014_spl_solmax")
weight_obj = simweights.GenieWeighter(simfile)
weights = weight_obj.get_weights(flux_model)
print("Rate", weights.sum(), "Hz")
