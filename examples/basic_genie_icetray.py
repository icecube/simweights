#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause
import nuflux
import numpy as np
import pandas as pd

import simweights

simfile = pd.HDFStore("level2_genie-icetray.140000_000000.hdf")
flux_model = nuflux.makeFlux("IPhonda2014_spl_solmax")
weight_obj = simweights.GenieWeighter(simfile, 1)
weights = weight_obj.get_weights(flux_model)
print("Rate from simweights", weights.sum(), "Hz")
simfile.close()
