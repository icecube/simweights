"""
Pure python library for calculating the weights of Monte Carlo simulation for IceCube.

SimWeights was designed with goal of calculating weights for IceCube simulation in a way that it
is easy to combine combine datasets with different generation parameters into a single sample.
It was also designed to be a stand alone project which does not depend on IceTray in any way so that it can
be installed easily on laptops. SimWeights gathers all the information it needs form information in the
hdf5 file so there is no need for access to the simulation production database. SimWeights works with
files produced with corsika-reader, neutrino-generator, and genie-reader.
"""

__version__ = "0.1"

__all__ = [
    "CircleInjector",
    "CorsikaWeighter",
    "NaturalRateCylinder",
    "UniformSolidAngleCylinder",
    "TIG1996",
    "FixedFractionFlux",
    "GenieWeighter",
    "GaisserH3a",
    "GaisserH4a",
    "GaisserH4a_IT",
    "GaisserHillas",
    "GlobalFitGST",
    "Hoerandel",
    "Hoerandel5",
    "Hoerandel_IT",
    "Honda2004",
    "NullSurface",
    "PDGCode",
    "corsika_to_pdg",
    "generation_surface",
    "GenerationSurfaceCollection",
    "NuGenWeighter",
    "PowerLaw",
    "Weighter",
]

from .corsika_weighter import CorsikaWeighter
from .fluxes import (
    TIG1996,
    FixedFractionFlux,
    GaisserH3a,
    GaisserH4a,
    GaisserH4a_IT,
    GaisserHillas,
    GlobalFitGST,
    Hoerandel,
    Hoerandel5,
    Hoerandel_IT,
    Honda2004,
)
from .generation_surface import GenerationSurfaceCollection, NullSurface, generation_surface
from .genie_weighter import GenieWeighter
from .nugen_weighter import NuGenWeighter
from .pdgcode import PDGCode
from .powerlaw import PowerLaw
from .spatial import CircleInjector, NaturalRateCylinder, UniformSolidAngleCylinder
from .utils import corsika_to_pdg
from .weighter import Weighter
