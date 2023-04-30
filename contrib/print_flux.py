#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from sys import argv

import numpy as np
from simweights import _fluxes

E = np.logspace(2, 10, 9)

for idx_v, name in enumerate(argv[1:]):
    print(f'"{name}": {{')
    flux = getattr(_fluxes, name)()

    for idx, i in enumerate(flux.pdgids):
        vals = [str(val * 10_000) for val in flux(E, i)]
        print(f'"{i}": [')
        print(f"  {', '.join(vals[:3])},")
        print(f"  {', '.join(vals[3:6])},")
        print(f"  {', '.join(vals[-3:])}")
        if idx != len(flux.pdgids) - 1:
            print("],")
        else:
            print("]")

    if idx_v != len(argv):
        print("},")
    else:
        print("}")
