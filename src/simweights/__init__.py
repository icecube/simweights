# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

"""Pure python library for calculating the weights of Monte Carlo simulation for IceCube.

SimWeights was designed with goal of calculating weights for IceCube simulation in a way that it
is easy to combine combine datasets with different generation parameters into a single sample.
It was also designed to be a stand alone project which does not depend on IceTray in any way so that it can
be installed easily on laptops. SimWeights gathers all the information it needs form information in the
hdf5 file so there is no need for access to the simulation production database. SimWeights works with
files produced with corsika-reader, neutrino-generator, and genie-reader.
"""

__version__ = "0.1.3"

__all__ = [
    "TIG1996",
    "CircleInjector",
    "CompositeSurface",
    "CorsikaWeighter",
    "FixedFractionFlux",
    "GaisserH3a",
    "GaisserH4a",
    "GaisserH4a_IT",
    "GaisserHillas",
    "GenerationSurface",
    "GenieSurface",
    "GenieWeighter",
    "GlobalFitGST",
    "GlobalFitGST_IT",
    "GlobalSplineFit",
    "GlobalSplineFit5Comp",
    "GlobalSplineFit_IT",
    "Hoerandel",
    "Hoerandel5",
    "Hoerandel_IT",
    "Honda2004",
    "IceTopSurface",
    "IceTopWeighter",
    "NaturalRateCylinder",
    "NuGenSurface",
    "NuGenWeighter",
    "PDGCode",
    "PowerLaw",
    "UniformSolidAngleCylinder",
    "Weighter",
    "corsika_to_pdg",
]

from ._corsika_weighter import CorsikaWeighter
from ._fluxes import (
    TIG1996,
    FixedFractionFlux,
    GaisserH3a,
    GaisserH4a,
    GaisserH4a_IT,
    GaisserHillas,
    GlobalFitGST,
    GlobalFitGST_IT,
    GlobalSplineFit,
    GlobalSplineFit5Comp,
    GlobalSplineFit_IT,
    Hoerandel,
    Hoerandel5,
    Hoerandel_IT,
    Honda2004,
)
from ._generation_surface import CompositeSurface, GenerationSurface
from ._genie_weighter import GenieSurface, GenieWeighter
from ._icetop_weighter import IceTopSurface, IceTopWeighter
from ._nugen_weighter import NuGenSurface, NuGenWeighter
from ._pdgcode import PDGCode
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector, NaturalRateCylinder, UniformSolidAngleCylinder
from ._utils import corsika_to_pdg
from ._weighter import Weighter
